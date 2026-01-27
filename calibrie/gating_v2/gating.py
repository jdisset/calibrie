# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

from __future__ import annotations
import logging
from functools import reduce
from typing import Annotated, ClassVar, Literal

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from dracon import dracon_program, Arg
from pydantic import BaseModel, Field, computed_field

from calibrie.pipeline import Task, DiagnosticFigure
from calibrie.plots import make_symlog_ax, density_histogram2d
from calibrie.utils import Context, escape_name, add_calibration_metadata, load_to_df
from declair_ui import UI, ClientComputed, render
from declair_ui.client_runnable import client_runnable
from calibrie.fcs_utils import (
    escape_name,
    parse_fcs_data,
    parse_fcs_meta,
    _parse_header,
    _parse_text_segment,
    _parse_int,
    _bytes_from_pnb,
)
from declair_ui.layout import Cols, Panel, Rows, Sidebar, Slot, UIConfig
from declair_ui.widgets import ColorPicker, Dropdown, FileList, TreeList, Overlay, ToolTray, ViewDeck

log = logging.getLogger(__name__)

_gating_instance: "GatingTask | None" = None


def _get_channels() -> list[str]:
    if _gating_instance and _gating_instance.files:
        channels = sorted({ch for f in _gating_instance.files for ch in f.channels})
        return channels if channels else ["(no channels)"]
    return ["(no channels)"]


Chan = Annotated[str, UI(Dropdown(choices=_get_channels))]


def random_color():
    h, s, v = np.random.uniform(0, 1), np.random.uniform(0.4, 0.8), np.random.uniform(0.9, 1)
    return [int(c * 255) for c in mcolors.hsv_to_rgb((h, s, v))] + [255]


def points_in_polygon(vertices: list[tuple[float, float]], x: list[float], y: list[float]) -> object:
    import numpy as np
    if len(vertices) < 3:
        return np.zeros(len(x), dtype=bool)
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    xi, yi, xj, yj = v[:, 0], v[:, 1], np.roll(v[:, 0], -1), np.roll(v[:, 1], -1)
    dy = yj - yi
    cond = (yi[:, None] > y_arr) ^ (yj[:, None] > y_arr)
    t = np.where(
        dy[:, None] != 0, (y_arr - yi[:, None]) / dy[:, None] * (xj - xi)[:, None] + xi[:, None], 0
    )
    return (np.count_nonzero(cond & (x_arr < t), axis=0) & 1).astype(bool)


class GateStatistics(BaseModel):
    count: int = 0
    total: int = 0
    percent_of_total: float = 0.0
    parent_count: int = 0
    percent_of_parent: float = 0.0


class PlotSource(BaseModel):
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    label: str = ""


class PlotResult(BaseModel):
    main: dict = Field(default_factory=dict)
    gates: list[dict] = Field(default_factory=list)
    gate_index: int = -1
    gate: dict | None = None


@client_runnable(helpers=[points_in_polygon])
def compute_gate_statistics(
    vertices: list[tuple[float, float]],
    x: list[float],
    y: list[float],
    parent_mask: list[bool] | None = None,
) -> GateStatistics:
    n_total = len(x)
    if n_total == 0:
        return GateStatistics()
    gate_mask = points_in_polygon(vertices, x, y)
    if parent_mask is not None:
        parent_arr = np.asarray(parent_mask, dtype=bool)
        combined_mask, parent_count = gate_mask & parent_arr, int(parent_arr.sum())
    else:
        combined_mask, parent_count = gate_mask, n_total
    count = int(combined_mask.sum())
    return GateStatistics(
        count=count,
        total=n_total,
        percent_of_total=count / n_total * 100 if n_total else 0.0,
        parent_count=parent_count,
        percent_of_parent=count / parent_count * 100 if parent_count else 0.0,
    )


class DataFile(BaseModel):
    path: str
    point_count: int = 0
    visible: bool = True
    group: str | None = None
    channels: list[str] = Field(default_factory=list)
    preview_data: list[list[float]] | None = Field(default=None)
    data_packed: dict | None = Field(default=None)

    @computed_field
    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1] if "/" in self.path else self.path


@client_runnable(
    helpers=[escape_name, _parse_int, _parse_header, _parse_text_segment, parse_fcs_meta, _bytes_from_pnb, parse_fcs_data],
    packages=["numpy"],
)
def load_fcs_data(file_bytes: bytes, filename: str, report_progress=None) -> dict:
    result = parse_fcs_data(file_bytes, report_progress=report_progress)
    return {
        "path": filename,
        "point_count": int(result["point_count"]),
        "channels": result["channels"],
        "preview_data": None,
        "data_packed": result["data_packed"],
        "visible": True,
    }


load_fcs_data.supports_progress = True


def _get_parent_chain(gates: list[dict], gate_index: int | None) -> list[dict]:
    if not gates or gate_index is None or gate_index < 0 or gate_index >= len(gates):
        return []
    gate = gates[gate_index]
    by_name = {g.get("name"): g for g in gates if isinstance(g, dict) and g.get("name")}
    chain: list[dict] = []
    parent_name = gate.get("parent_gate")
    while parent_name:
        parent = by_name.get(parent_name)
        if not parent:
            break
        chain.append(parent)
        parent_name = parent.get("parent_gate")
    return chain


def _compute_parent_mask(parent_chain: list[dict], data: dict) -> object:
    import numpy as np
    if not parent_chain:
        return None
    n = len(next(iter(data.values())))
    mask = np.ones(n, dtype=bool)
    for p in parent_chain:
        if not p.get("applied", False):
            continue
        verts = p.get("vertices", [])
        if len(verts) < 3:
            continue
        px, py = data.get(p.get("xchannel")), data.get(p.get("ychannel"))
        if px is None or py is None:
            continue
        mask = mask & points_in_polygon(verts, px, py)
    return mask if not np.all(mask) else None


def _extract_binary_data(f: dict, required: list[str]) -> tuple[dict[str, object], bool]:
    import numpy as np
    channels = f.get("channels") or []
    if not channels:
        print(f"[_extract_binary_data] No channels in file")
        return {}, False
    idx_map = {ch: i for i, ch in enumerate(channels)}
    missing = [ch for ch in required if ch not in idx_map]
    if missing:
        print(f"[_extract_binary_data] Missing channels: {missing}, available: {channels[:5]}...")
        return {}, False

    arr = None
    binary = f.get("preview_data_binary")
    if isinstance(binary, np.ndarray) and binary.ndim == 2:
        arr = binary
        print(f"[_extract_binary_data] Using ndarray binary: shape={arr.shape}")
    elif isinstance(binary, dict):
        data, shape = binary.get("data"), binary.get("shape")
        print(f"[_extract_binary_data] Binary dict: data_type={type(data).__name__}, shape={shape}")
        if data is not None and shape and len(shape) == 2:
            try:
                arr = np.asarray(data, dtype=np.float32)
                expected = int(shape[0]) * int(shape[1])
                print(f"[_extract_binary_data] arr.ndim={arr.ndim}, arr.size={arr.size}, expected={expected}")
                if arr.ndim == 1 and arr.size >= expected:
                    arr = arr[:expected].reshape(int(shape[0]), int(shape[1]))
                    print(f"[_extract_binary_data] Reshaped to {arr.shape}")
            except Exception as e:
                print(f"[_extract_binary_data] Error reshaping: {e}")
                arr = None
    if arr is None:
        preview = f.get("preview_data")
        if preview:
            arr = np.asarray(preview, dtype=np.float32)
            print(f"[_extract_binary_data] Using preview_data: shape={arr.shape}")
            if arr.ndim != 2:
                arr = None
    if arr is None:
        print(f"[_extract_binary_data] No valid data found")
        return {}, False

    result = {}
    for ch in required:
        i = idx_map[ch]
        if i < arr.shape[1]:
            result[ch] = arr[:, i]
    print(f"[_extract_binary_data] Extracted {len(result)} channels, {len(result[list(result.keys())[0]]) if result else 0} points")
    return result, True


def _build_groups(files: list[dict], required: set[str]) -> dict[str, dict[str, object]]:
    import numpy as np
    buffers: dict[str, dict[str, list]] = {}
    for f in files:
        if not f.get("visible", True):
            continue
        data, ok = _extract_binary_data(f, list(required))
        if not ok:
            continue
        group = f.get("group") or "All Files"
        buf = buffers.setdefault(group, {ch: [] for ch in required})
        for ch in required:
            if ch in data:
                buf[ch].append(data[ch])

    groups: dict[str, dict[str, object]] = {}
    for gname, bufs in buffers.items():
        groups[gname] = {}
        for ch, chunks in bufs.items():
            if chunks:
                groups[gname][ch] = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return groups


def _compute_sources(groups: dict, xaxis: str, yaxis: str, parent_chain: list[dict]) -> dict:
    sources: list[dict] = []
    if not xaxis or not yaxis:
        return {"sources": sources}
    for gname, data in groups.items():
        if xaxis not in data or yaxis not in data:
            continue
        x_all, y_all = data[xaxis], data[yaxis]
        if len(x_all) == 0:
            continue
        mask = _compute_parent_mask(parent_chain, data) if parent_chain else None
        if mask is not None:
            x_all, y_all = x_all[mask], y_all[mask]
        sources.append({"x": x_all.tolist(), "y": y_all.tolist(), "label": gname})
    return {"sources": sources}


def _gather_required_channels(xaxis: str, yaxis: str, gates: list[dict], mode: str, selected: int) -> set[str]:
    required: set[str] = set()
    if xaxis and yaxis:
        required.update([xaxis, yaxis])

    if mode == "deck":
        for i, g in enumerate(gates):
            gx, gy = g.get("xchannel"), g.get("ychannel")
            if gx and gy:
                required.update([gx, gy])
            for p in _get_parent_chain(gates, i):
                if p.get("applied"):
                    px, py = p.get("xchannel"), p.get("ychannel")
                    if px and py:
                        required.update([px, py])
    elif mode == "active" and 0 <= selected < len(gates):
        g = gates[selected]
        gx, gy = g.get("xchannel"), g.get("ychannel")
        if gx and gy:
            required.update([gx, gy])
        for p in _get_parent_chain(gates, selected):
            if p.get("applied"):
                px, py = p.get("xchannel"), p.get("ychannel")
                if px and py:
                    required.update([px, py])
    return required


@client_runnable(helpers=[points_in_polygon, _get_parent_chain, _compute_parent_mask, _extract_binary_data, _build_groups, _compute_sources, _gather_required_channels])
def compute_plot_data(
    files: list[dict],
    xaxis: str,
    yaxis: str,
    gates: list[dict] | None = None,
    selected_gate_index: int | None = None,
    mode: Literal["deck", "active"] = "deck",
) -> dict:
    print(f"[compute_plot_data] mode={mode}, xaxis={xaxis}, yaxis={yaxis}, files={len(files) if files else 0}, gates={len(gates) if gates else 0}")
    gates = gates or []
    if not files:
        print("[compute_plot_data] No files, returning empty")
        return {"main": {}, "gates": [], "gate_index": -1, "gate": None}

    for i, f in enumerate(files[:3]):
        print(f"[compute_plot_data] file[{i}]: visible={f.get('visible')}, channels={len(f.get('channels') or [])}, has_binary={f.get('preview_data_binary') is not None}, has_preview={f.get('preview_data') is not None}")

    selected = int(selected_gate_index) if selected_gate_index is not None else -1
    required = _gather_required_channels(xaxis, yaxis, gates, mode, selected)
    print(f"[compute_plot_data] required_channels={required}")
    if not required:
        print("[compute_plot_data] No required channels, returning empty")
        return {"main": {}, "gates": [], "gate_index": -1, "gate": None}

    groups = _build_groups(files, required)
    print(f"[compute_plot_data] groups={list(groups.keys())}, group_sizes={{k: {{ch: len(v) for ch, v in d.items()}} for k, d in groups.items()}}")
    if not groups:
        print("[compute_plot_data] No groups built, returning empty")
        return {"main": {}, "gates": [], "gate_index": -1, "gate": None}

    if mode == "deck":
        main = _compute_sources(groups, xaxis, yaxis, [])
        print(f"[compute_plot_data] deck mode: main has {len(main.get('sources', []))} sources")
        if main.get('sources'):
            s = main['sources'][0]
            print(f"[compute_plot_data] first source: {len(s.get('x', []))} points, label={s.get('label')}")
        gate_sources = []
        for i, g in enumerate(gates):
            gx, gy = g.get("xchannel"), g.get("ychannel")
            if gx and gy:
                gate_sources.append(_compute_sources(groups, gx, gy, _get_parent_chain(gates, i)))
            else:
                gate_sources.append({"sources": []})
        result = {"main": main, "gates": gate_sources, "gate_index": -1, "gate": None}
        print(f"[compute_plot_data] returning deck result with {len(gate_sources)} gate sources")
        return result

    if 0 <= selected < len(gates):
        g = gates[selected]
        gx, gy = g.get("xchannel"), g.get("ychannel")
        if gx and gy:
            gate_data = _compute_sources(groups, gx, gy, _get_parent_chain(gates, selected))
            print(f"[compute_plot_data] active mode with gate: {len(gate_data.get('sources', []))} sources")
            return {"main": {}, "gates": [], "gate_index": selected, "gate": gate_data}
    main = _compute_sources(groups, xaxis, yaxis, [])
    print(f"[compute_plot_data] active mode no gate: main has {len(main.get('sources', []))} sources")
    return {"main": main, "gates": [], "gate_index": -1, "gate": None}


class PolygonGate(BaseModel):
    gate_type: Literal["Polygon"] = "Polygon"
    name: str = "Gate"
    gate_mode: Annotated[Literal["navigation", "polygon_gate"], UI(hidden=True)] = "navigation"
    xchannel: Chan = ""
    ychannel: Chan = ""
    vertices: Annotated[
        list[tuple[float, float]],
        UI(
            Overlay(
                selection_field="selected_gate_index",
                color_field="color",
                mode_field="gate_mode",
                active_mode_value="polygon_gate",
                stats_function="compute_gate_statistics",
                stats_primary_field="count",
                stats_secondary_field="percent_of_parent",
                stats_primary_label="events",
                stats_secondary_label="% of parent",
                render_only_when_selected=True,
                insert_mode="click",
            ),
            uses="active-plot",
        ),
    ] = Field(default_factory=list)
    color: Annotated[tuple[int, int, int, int], UI(ColorPicker(no_inputs=True), inline_with="name")] = (100, 200, 255, 200)
    applied: Annotated[bool, UI(hidden=True)] = False
    parent_gate: str | None = None
    event_count: int = Field(default=0, exclude=True)
    percent_of_parent: float = Field(default=0.0, exclude=True)
    stats_summary: str = Field(default="No data", exclude=True)

    def contains_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return points_in_polygon(list(self.vertices), x, y)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.vertices) < 3:
            return df
        return df[self.contains_points(df[self.xchannel].values, df[self.ychannel].values)]


class GatingTask(Task):
    gates: Annotated[
        list[PolygonGate],
        UI(TreeList(
            parent_field="parent_gate",
            badge_field="event_count",
            color_field="color",
            selection_field="selected_gate_index",
            apply_field="applied",
            tooltip_field="stats_summary",
            row2_fields=["xchannel", "ychannel"],
        )),
    ] = Field(default_factory=list)
    target_population: str | None = None

    files: Annotated[
        list[DataFile],
        UI(FileList(
            file_processor=load_fcs_data,
            badge_field="point_count",
            visibility_field="visible",
            cache_fields=["data_packed"],
            group_field="group",
            group_label="Group",
            group_mode="sections",
            ungrouped_label="All Files",
            file_extensions=[".fcs"],
        )),
    ] = Field(default_factory=list, exclude=True)
    xaxis: Annotated[Chan, UI(label="X Channel")] = Field(default="", exclude=True)
    yaxis: Annotated[Chan, UI(label="Y Channel")] = Field(default="", exclude=True)
    gate_tools: Annotated[
        dict,
        UI(ToolTray(
            items_field="gates",
            selection_field="selected_gate_index",
            type_field="gate_type",
            type_choices=["Polygon"],
            plot_ref="active-plot",
            stats_function="compute_gate_statistics",
            stats_primary_field="count",
            stats_secondary_field="percent_of_parent",
            stats_primary_label="events",
            stats_secondary_label="% of parent",
        )),
    ] = Field(default_factory=dict, exclude=True)
    plot_data: Annotated[
        dict,
        UI(ViewDeck(
            selection_field="selected_gate_index",
            items_field="gates",
            label_field="name",
            prepend_tab="Main View",
            disable_inactive=True,
            content_config={"width": 800, "height": 600, "default_x_transform": "symlog", "default_y_transform": "symlog"},
        )),
        ClientComputed(
            compute=compute_plot_data,
            depends_on=["files", "xaxis", "yaxis", "gates"],
            preparation={
                "type": "channel",
                "channelFields": ["xaxis", "yaxis"],
                "itemsField": "gates",
                "selectionField": "selected_gate_index",
                "mode": "deck",
            },
        ),
    ] = Field(default_factory=dict, exclude=True)
    selected_gate_index: int = Field(default=-1, exclude=True)

    ui_config: ClassVar[UIConfig] = UIConfig(
        bottom_bar=True,
        layout=Cols(
            Sidebar("files", "gates", width=300),
            Rows(
                Slot("gates", widget_override="list-editor", config_override={
                    "compact": True,
                    "selectable": True,
                    "tab_style": True,
                    "display_field": "name",
                    "color_field": "color",
                    "selection_field": "selected_gate_index",
                    "prepend_tab": {"label": "Main View", "index": -1},
                    "add_mode": "dialog",
                    "edit_mode": "dialog",
                    "auto_name": {"field": "name", "prefix": "Gate", "start": 1},
                    "auto_color": {"field": "color", "source_field": "name", "method": "hash-hsv", "saturation": 0.8, "value": 0.8, "alpha": 200, "overwrite": True},
                    "inherit_fields": [{"from": "xaxis", "to": "xchannel"}, {"from": "yaxis", "to": "ychannel"}],
                    "exclude_fields": ["gate_mode", "applied"],
                    "dialog_label": "Gate",
                }),
                Panel("gate_tools"),
                Panel("plot_data", overlays=[("gates", "vertices", "selected_gate_index >= 0")]),
            ),
        ),
    )
    client_functions: ClassVar = [compute_gate_statistics]

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        global _gating_instance
        _gating_instance = self
        log.debug("[gating_v2] GatingTask initialized with %d files, %d gates", len(self.files), len(self.gates))
        self._recompute_stats()

    def initialize(self, ctx: Context) -> Context:
        return Context(cell_data_loader=self.load_cells)

    def load_cells(self, data, column_order: list[str] | None = None) -> pd.DataFrame:
        df = load_to_df(data)
        prev_n = len(df)
        log.debug("[gating_v2] load_cells: input %d cells, target_population=%s", prev_n, self.target_population)
        if self.target_population:
            self._validate_hierarchy()
            data_dict = {col: df[col].values for col in df.columns}
            df = df[self.get_population_mask(self.target_population, data_dict)]
        else:
            df = reduce(lambda d, g: g(d), self.gates, df)
        add_calibration_metadata(df, self._name, {"n_before": prev_n, "n_after": len(df), "target": self.target_population})
        log.debug("[gating_v2] load_cells: %d -> %d cells after gating", prev_n, len(df))
        return df[column_order] if column_order else df

    def get_gate_by_name(self, name: str) -> PolygonGate | None:
        return next((g for g in self.gates if g.name == name), None)

    def get_population_mask(self, name: str | None, data: dict[str, np.ndarray]) -> np.ndarray:
        if name is None:
            return np.ones(len(next(iter(data.values()))), dtype=bool)
        gate = self.get_gate_by_name(name)
        assert gate, f"Unknown gate: {name}"
        return self.get_population_mask(gate.parent_gate, data) & gate.contains_points(data[gate.xchannel], data[gate.ychannel])

    def _validate_hierarchy(self) -> None:
        names = {g.name for g in self.gates}
        bad = [g.name for g in self.gates if g.parent_gate and g.parent_gate not in names]
        assert not bad, f"Gates with unknown parents: {bad}"

    def _get_all_data(self) -> dict[str, np.ndarray]:
        all_data: dict[str, np.ndarray] = {}
        for f in self.files:
            if not f.visible or not f.preview_data or not f.channels:
                continue
            arr = np.array(f.preview_data)
            if arr.ndim == 2 and arr.shape[1] == len(f.channels):
                for i, ch in enumerate(f.channels):
                    all_data[ch] = np.concatenate([all_data[ch], arr[:, i]]) if ch in all_data else arr[:, i]
        return all_data

    def _recompute_stats(self) -> None:
        all_data = self._get_all_data()
        log.debug("[gating_v2] _recompute_stats: %d channels, %d gates", len(all_data), len(self.gates))
        for gate in self.gates:
            try:
                if not all_data:
                    gate.event_count, gate.percent_of_parent, gate.stats_summary = 0, 0.0, "No data"
                    continue
                count = int(self.get_population_mask(gate.name, all_data).sum())
                parent_count = int(self.get_population_mask(gate.parent_gate, all_data).sum()) if gate.parent_gate else len(next(iter(all_data.values())))
                pct = count / parent_count * 100 if parent_count else 0.0
                gate.event_count, gate.percent_of_parent, gate.stats_summary = count, pct, f"{count:,} ({pct:.1f}%)"
                log.debug("[gating_v2] gate %s: %d/%d (%.1f%%)", gate.name, count, parent_count, pct)
            except (KeyError, ValueError, AssertionError) as e:
                log.debug("[gating_v2] gate %s: stats failed: %s", gate.name, e)
                gate.event_count, gate.percent_of_parent, gate.stats_summary = 0, 0.0, "N/A"

    def diagnostics(
        self, ctx: Context, use_files: list | None = None, resample_to: int = 100000, nbins: int = 400,
        density_lims: tuple = (1e-6, None), figsize: tuple = (12, 12), dpi: int = 200, **_,
    ) -> list[DiagnosticFigure]:
        from matplotlib import pyplot as plt
        src = ctx.raw_color_controls.values() if use_files is None else [load_to_df(f) for f in use_files]
        df = pd.concat(src).sample(frac=1).iloc[:resample_to]
        gates_by_axes: dict[tuple, list[PolygonGate]] = {}
        for g in self.gates:
            if len(g.vertices) > 2:
                gates_by_axes.setdefault(tuple(sorted([g.xchannel, g.ychannel])), []).append(g)
        if not gates_by_axes:
            return []
        n = len(gates_by_axes)
        nc = min(max(int(np.ceil(np.sqrt(n))), 1), 3)
        nr = max(int(np.ceil(n / nc)), 1)
        fig, axes = plt.subplots(nr, nc, figsize=figsize, dpi=dpi)
        flat_axes = np.atleast_1d(axes).flatten()
        for i, ((xax, yax), glist) in enumerate(gates_by_axes.items()):
            ax = flat_axes[i]
            X = df[[xax, yax]].values
            tr, _, xlims_tr, ylims_tr = make_symlog_ax(ax, (X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max()))
            Xtr = tr(X)
            density_histogram2d(ax, Xtr[:, 0], Xtr[:, 1], xlims_tr, ylims_tr, nbins, vmin=density_lims[0], vmax=density_lims[1], noise_smooth=0.01)
            for g in glist:
                col = [c / 255 for c in random_color()]
                verts = tr(np.vstack([g.vertices, g.vertices[0]])) if g.vertices else np.array([])
                x, y = (verts.T if g.xchannel == xax else verts.T[::-1]) if len(verts) else ([], [])
                after = g(df)
                ax.plot(x, y, color=col, lw=1.5, label=f"{g.name}: {len(after)}/{len(df)} ({len(after) / len(df) * 100:.1f}%)")
                ax.fill(x, y, color=col, alpha=0.2)
            ax.axvline(0, c="k", lw=0.5, alpha=0.5, ls="--")
            ax.axhline(0, c="k", lw=0.5, alpha=0.5, ls="--")
            ax.legend()
            ax.set_aspect("equal")
            ax.set_xlabel(xax)
            ax.set_ylabel(yax)
        final = reduce(lambda d, g: g(d), self.gates, df)
        fig.suptitle(f"After gating: {len(final)}/{len(df)} ({len(final) / len(df) * 100:.1f}%)")
        fig.tight_layout(pad=2)
        return [DiagnosticFigure(fig, "Gating")]


GatingSession = GatingTask


@dracon_program(
    name='calibrie-gating-v2',
    description='Interactive polygon gating for flow cytometry data',
    context_types=[PolygonGate, DataFile],
)
class GatingConfig(BaseModel):
    """Configuration for calibrie-gating-v2 CLI."""
    verbose: Annotated[bool, Arg(short='v', help="Enable debug logging")] = False
    browser: Annotated[bool, Arg(short='b', help="Open in browser instead of native window")] = False
    port: int = 8080
    host: str = "127.0.0.1"

    def run(self) -> GatingTask:
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

        if self.verbose:
            logging.getLogger("calibrie").setLevel(logging.DEBUG)
            logging.getLogger("declair_ui").setLevel(logging.DEBUG)
            log.debug("[calibrie-gating-v2] Debug mode enabled")

        backend: Literal["browser", "native"] = "browser" if self.browser else "native"

        if backend == "native":
            try:
                import webview  # noqa: F401
            except ImportError:
                log.warning("pywebview not installed, using browser. Install: pip install 'calibrie[native]'")
                backend = "browser"

        log.debug("[calibrie-gating-v2] Starting with backend=%s, port=%d, host=%s", backend, self.port, self.host)
        task = GatingTask()
        result = render(
            task,
            title="Calibrie Gating",
            channels_handler=_get_channels,
            backend=backend,
            debug=self.verbose,
            port=self.port,
            host=self.host,
        )
        if result:
            log.info("Session: %d files, %d gates", len(result.files), len(result.gates))
        return result


def main(backend: Literal["browser", "native"] = "native", verbose: bool = False) -> GatingTask:
    """Legacy entry point for programmatic usage."""
    return GatingConfig(browser=(backend == "browser"), verbose=verbose).run()


if __name__ == "__main__":
    GatingConfig.cli()
