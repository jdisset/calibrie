# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

from __future__ import annotations
import logging
import sys
from functools import reduce
from typing import Annotated, ClassVar, Literal

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
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
from declair_ui.widgets import ColorPicker, DensityPlot, Dropdown, FileList, GateToolTray, PolygonDrawer, TreeList

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


def points_in_polygon(
    vertices: list[tuple[float, float]], x: list[float], y: list[float]
) -> object:
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
    helpers=[
        escape_name,
        _parse_int,
        _parse_header,
        _parse_text_segment,
        parse_fcs_meta,
        _bytes_from_pnb,
        parse_fcs_data,
    ],
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


#TODO: shorter simpler code?
@client_runnable(helpers=[points_in_polygon])
def compute_plot_data(
    files: list[dict],
    xaxis: str,
    yaxis: str,
    gates: list[dict] | None = None,
    selected_gate_index: int | None = None,
) -> dict:
    """Compute multi-source plot data client-side from cached file data.

    Files are grouped by their 'group' field. Files without a group are
    merged into a single "All Files" source.

    Supports hierarchical filtering: when a gate is selected, only events
    passing through applied parent gates are shown.

    Supports both binary transfer (preview_data_binary with numpy array) and
    legacy JSON transfer (preview_data with nested arrays).

    Display resampling is handled by the UI layer (perSourceMaxPoints).
    """
    import numpy as np

    def compute_parent_chain_mask(
        gates_list: list[dict],
        gate: dict,
        all_data: dict,
    ) -> object:
        """Walk parent chain, apply masks for gates with applied=True."""
        gate_by_name = {g["name"]: g for g in gates_list}
        n_points = len(next(iter(all_data.values())))
        mask = np.ones(n_points, dtype=bool)

        parent_name = gate.get("parent_gate")
        while parent_name:
            parent = gate_by_name.get(parent_name)
            if not parent:
                break
            if parent.get("applied", False):
                vertices = parent.get("vertices", [])
                if len(vertices) >= 3:
                    px = all_data.get(parent["xchannel"])
                    py = all_data.get(parent["ychannel"])
                    if px is not None and py is not None:
                        parent_in = points_in_polygon(vertices, px, py)
                        mask = mask & parent_in
            parent_name = parent.get("parent_gate")

        return mask if not np.all(mask) else None

    active_xaxis = xaxis
    active_yaxis = yaxis

    if gates is None:
        gates = []

    if selected_gate_index is not None and selected_gate_index >= 0:
        if selected_gate_index < len(gates):
            gate = gates[selected_gate_index]
            gate_x = gate.get("xchannel")
            gate_y = gate.get("ychannel")
            if gate_x and gate_y:
                active_xaxis = gate_x
                active_yaxis = gate_y

    if not active_xaxis or not active_yaxis:
        return {}
    if not files:
        return {}

    def file_signature(file_list: list[dict]) -> tuple:
        sig: list[tuple] = []
        for f in file_list:
            if not f or not isinstance(f, dict):
                sig.append(("?", False, (), 0))
                continue
            path = f.get("path") or f.get("filename") or ""
            visible = bool(f.get("visible", True))
            channels = tuple(f.get("channels") or ())
            binary = f.get("preview_data_binary")
            if isinstance(binary, np.ndarray) and binary.ndim == 2:
                n = int(binary.shape[0])
            else:
                data = f.get("preview_data")
                n = len(data) if isinstance(data, list) else 0
            sig.append((path, visible, channels, n))
        return tuple(sig)

    global _calibrie_plot_cache
    try:
        _calibrie_plot_cache
    except NameError:
        _calibrie_plot_cache = {}

    files_key = file_signature(files)
    cache_key = (files_key, active_xaxis, active_yaxis)
    cache = _calibrie_plot_cache.get("data")
    if not cache or cache.get("key") != cache_key:
        chunks_x: list[np.ndarray] = []
        chunks_y: list[np.ndarray] = []
        for f in files:
            if not f.get("visible", True) or not f.get("channels"):
                continue
            channels = f["channels"]

            binary = f.get("preview_data_binary")
            if isinstance(binary, np.ndarray) and binary.ndim == 2:
                arr = binary
            else:
                arr = None

            if arr is None:
                data = f.get("preview_data")
                if not data:
                    continue
                arr = np.asarray(data, dtype=np.float32)
                if arr.ndim != 2:
                    continue

            if len(channels) >= 2 and channels[0] == active_xaxis and channels[1] == active_yaxis and arr.shape[1] >= 2:
                xi, yi = 0, 1
            else:
                try:
                    xi = channels.index(active_xaxis)
                    yi = channels.index(active_yaxis)
                except ValueError:
                    continue

            if xi >= arr.shape[1] or yi >= arr.shape[1]:
                continue

            chunks_x.append(arr[:, xi])
            chunks_y.append(arr[:, yi])

        if not chunks_x:
            return {}

        total = sum(int(c.shape[0]) for c in chunks_x)
        x_all = np.empty(total, dtype=np.float32)
        y_all = np.empty(total, dtype=np.float32)
        offset = 0
        for cx, cy in zip(chunks_x, chunks_y):
            n = int(cx.shape[0])
            if n == 0:
                continue
            x_all[offset : offset + n] = cx
            y_all[offset : offset + n] = cy
            offset += n
        if offset != total:
            x_all = x_all[:offset]
            y_all = y_all[:offset]

        cache = {
            "key": cache_key,
            "x_all": x_all,
            "y_all": y_all,
            "x_list": x_all.tolist(),
            "y_list": y_all.tolist(),
            "mask_cache": {},
        }
        _calibrie_plot_cache["data"] = cache

    x_all = cache["x_all"]
    y_all = cache["y_all"]
    if len(x_all) == 0:
        return {}

    all_data = {active_xaxis: x_all, active_yaxis: y_all}
    label = "Filtered" if (gates and selected_gate_index is not None and selected_gate_index >= 0) else "All Files"

    if gates and selected_gate_index is not None and 0 <= selected_gate_index < len(gates):
        gate = gates[selected_gate_index]
        gate_by_name = {g["name"]: g for g in gates}

        chain_sig: list[tuple] = []
        parent_name = gate.get("parent_gate")
        while parent_name:
            parent = gate_by_name.get(parent_name)
            if not parent:
                break
            chain_sig.append(
                (
                    parent.get("name"),
                    bool(parent.get("applied", False)),
                    parent.get("xchannel"),
                    parent.get("ychannel"),
                    tuple(tuple(v) for v in (parent.get("vertices") or ())),
                )
            )
            parent_name = parent.get("parent_gate")
        chain_sig_t = tuple(chain_sig)

        mask_cache = cache.get("mask_cache", {})
        mask_key = (cache_key, chain_sig_t)
        cached_mask = mask_cache.get(mask_key)
        if cached_mask is not None:
            if cached_mask is False:
                return {"sources": [{"x": cache["x_list"], "y": cache["y_list"], "label": label}]}
            x_list, y_list = cached_mask
            return {"sources": [{"x": x_list, "y": y_list, "label": label}]}

        parent_mask = compute_parent_chain_mask(gates, gate, all_data)
        if parent_mask is None or len(parent_mask) != len(x_all):
            mask_cache[mask_key] = False
            cache["mask_cache"] = mask_cache
            return {"sources": [{"x": cache["x_list"], "y": cache["y_list"], "label": label}]}

        x_f = x_all[parent_mask]
        y_f = y_all[parent_mask]
        x_list = x_f.tolist()
        y_list = y_f.tolist()
        mask_cache[mask_key] = (x_list, y_list)
        cache["mask_cache"] = mask_cache
        return {"sources": [{"x": x_list, "y": y_list, "label": label}]}

    return {"sources": [{"x": cache["x_list"], "y": cache["y_list"], "label": label}]}


class PolygonGate(BaseModel):
    gate_type: Literal["Polygon"] = "Polygon"
    name: str = "Gate"
    gate_mode: Annotated[
        Literal["navigation", "polygon_gate"],
        UI(hidden=True),
    ] = "navigation"
    xchannel: Chan = ""
    ychannel: Chan = ""
    vertices: Annotated[
        list[tuple[float, float]],
        UI(
            PolygonDrawer(
                selection_field="selected_gate_index",
                color_field="color",
                stats_function="compute_gate_statistics",
                stats_primary_field="count",
                stats_secondary_field="percent_of_parent",
                stats_primary_label="events",
                stats_secondary_label="% of parent",
            ),
            uses="main-plot",
        ),
    ] = Field(default_factory=list)
    color: Annotated[tuple[int, int, int, int], UI(ColorPicker())] = (100, 200, 255, 200)
    applied: bool = False
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
    """Polygon gating task - both a pipeline Task AND a declair-UI renderable model."""

    gates: Annotated[
        list[PolygonGate],
        UI(
            TreeList(
                parent_field="parent_gate",
                badge_field="event_count",
                color_field="color",
                selection_field="selected_gate_index",
                apply_field="applied",
                tooltip_field="stats_summary",
                row2_fields=["xchannel", "ychannel"],
            )
        ),
    ] = Field(default_factory=list)
    target_population: str | None = None

    files: Annotated[
        list[DataFile], UI(FileList(
            file_processor=load_fcs_data,
            badge_field="point_count",
            visibility_field="visible",
            cache_fields=["data_packed"],
            group_field="group",
            group_label="Group",
            group_mode="sections",
            ungrouped_label="All Files",
            file_extensions=[".fcs"],
        ))
    ] = Field(default_factory=list, exclude=True)
    xaxis: Annotated[Chan, UI(label="X Channel")] = Field(default="", exclude=True)
    yaxis: Annotated[Chan, UI(label="Y Channel")] = Field(default="", exclude=True)
    gate_tools: Annotated[
        dict,
        UI(
            GateToolTray(
                gates_field="gates",
                selection_field="selected_gate_index",
                gate_type_field="gate_type",
                gate_type_choices=["Polygon"],
                plot_ref="main-plot",
            )
        ),
    ] = Field(default_factory=dict, exclude=True)
    plot_data: Annotated[
        dict,
        UI(
            DensityPlot(
                width=800,
                height=600,
                default_x_transform="symlog",
                default_y_transform="symlog",
                x_axis_slot="xaxis",
                y_axis_slot="yaxis",
                axis_slot_editable_field="selected_gate_index",
                axis_slot_editable_value=-1,
                view_state_key_field="selected_gate_index",
            ),
            provides="main-plot",
        ),
        ClientComputed(compute=compute_plot_data, depends_on=["files", "xaxis", "yaxis", "gates", "selected_gate_index"], preparation="flow-cytometry"),
    ] = Field(default_factory=dict, exclude=True)
    selected_gate_index: int = Field(default=-1, exclude=True)

    ui_config: ClassVar[UIConfig] = UIConfig(
        bottom_bar=True,
        layout=Cols(
            Sidebar("files", "gates", width=300),
            Rows(
                Slot(
                    "gates",
                    widget_override="list-editor",
                    config_override={
                        "compact": True,
                        "selectable": True,
                        "tab_style": True,
                        "display_field": "name",
                        "color_field": "color",
                        "selection_field": "selected_gate_index",
                        "prepend_tab": {"label": "Main View", "index": -1},
                    },
                ),
                Panel("gate_tools"),
                Panel("plot_data", overlays=[("gates", "vertices", "selected_gate_index >= 0")]),
            ),
        )
    )
    client_functions: ClassVar = [compute_gate_statistics]

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        global _gating_instance
        _gating_instance = self
        self._recompute_stats()

    def initialize(self, ctx: Context) -> Context:
        return Context(cell_data_loader=self.load_cells)

    def load_cells(self, data, column_order: list[str] | None = None) -> pd.DataFrame:
        df = load_to_df(data)
        prev_n = len(df)
        if self.target_population:
            self._validate_hierarchy()
            data_dict = {col: df[col].values for col in df.columns}
            df = df[self.get_population_mask(self.target_population, data_dict)]
        else:
            df = reduce(lambda d, g: g(d), self.gates, df)
        add_calibration_metadata(
            df,
            self._name,
            {"n_before": prev_n, "n_after": len(df), "target": self.target_population},
        )
        print(f"LOADER--gating: {prev_n} -> {len(df)} cells")
        return df[column_order] if column_order else df

    def get_gate_by_name(self, name: str) -> PolygonGate | None:
        return next((g for g in self.gates if g.name == name), None)

    def get_population_mask(self, name: str | None, data: dict[str, np.ndarray]) -> np.ndarray:
        if name is None:
            return np.ones(len(next(iter(data.values()))), dtype=bool)
        gate = self.get_gate_by_name(name)
        assert gate, f"Unknown gate: {name}"
        return self.get_population_mask(gate.parent_gate, data) & gate.contains_points(
            data[gate.xchannel], data[gate.ychannel]
        )

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
                    all_data[ch] = (
                        np.concatenate([all_data[ch], arr[:, i]]) if ch in all_data else arr[:, i]
                    )
        return all_data

    def _recompute_stats(self) -> None:
        all_data = self._get_all_data()
        for gate in self.gates:
            try:
                if not all_data:
                    gate.event_count, gate.percent_of_parent, gate.stats_summary = 0, 0.0, "No data"
                    continue
                count = int(self.get_population_mask(gate.name, all_data).sum())
                parent_count = (
                    int(self.get_population_mask(gate.parent_gate, all_data).sum())
                    if gate.parent_gate
                    else len(next(iter(all_data.values())))
                )
                pct = count / parent_count * 100 if parent_count else 0.0
                gate.event_count, gate.percent_of_parent, gate.stats_summary = (
                    count,
                    pct,
                    f"{count:,} ({pct:.1f}%)",
                )
            except (KeyError, ValueError, AssertionError):
                gate.event_count, gate.percent_of_parent, gate.stats_summary = 0, 0.0, "N/A"

    def diagnostics(
        self,
        ctx: Context,
        use_files: list | None = None,
        resample_to: int = 100000,
        nbins: int = 400,
        density_lims: tuple = (1e-6, None),
        figsize: tuple = (12, 12),
        dpi: int = 200,
        **_,
    ) -> list[DiagnosticFigure]:
        from matplotlib import pyplot as plt

        src = (
            ctx.raw_color_controls.values()
            if use_files is None
            else [load_to_df(f) for f in use_files]
        )
        df = pd.concat(src).sample(frac=1).iloc[:resample_to]

        gates_by_axes: dict[tuple, list[PolygonGate]] = {}
        for g in self.gates:
            if len(g.vertices) > 2:
                gates_by_axes.setdefault(tuple(sorted([g.xchannel, g.ychannel])), []).append(g)
        if not gates_by_axes:
            return []

        n = len(gates_by_axes)
        nc, nr = (
            min(max(int(np.ceil(np.sqrt(n))), 1), 3),
            max(int(np.ceil(n / min(max(int(np.ceil(np.sqrt(n))), 1), 3))), 1),
        )
        fig, axes = plt.subplots(nr, nc, figsize=figsize, dpi=dpi)
        flat_axes = np.atleast_1d(axes).flatten()

        for i, ((xax, yax), glist) in enumerate(gates_by_axes.items()):
            ax = flat_axes[i]
            X = df[[xax, yax]].values
            tr, _, xlims_tr, ylims_tr = make_symlog_ax(
                ax, (X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max())
            )
            Xtr = tr(X)
            density_histogram2d(
                ax,
                Xtr[:, 0],
                Xtr[:, 1],
                xlims_tr,
                ylims_tr,
                nbins,
                vmin=density_lims[0],
                vmax=density_lims[1],
                noise_smooth=0.01,
            )
            for g in glist:
                col = [c / 255 for c in random_color()]
                verts = tr(np.vstack([g.vertices, g.vertices[0]])) if g.vertices else np.array([])
                x, y = (verts.T if g.xchannel == xax else verts.T[::-1]) if len(verts) else ([], [])
                after = g(df)
                ax.plot(
                    x,
                    y,
                    color=col,
                    lw=1.5,
                    label=f"{g.name}: {len(after)}/{len(df)} ({len(after) / len(df) * 100:.1f}%)",
                )
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


def main(backend: Literal["browser", "native"] = "native") -> GatingTask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if backend == "native":
        try:
            import webview  # noqa: F401
        except ImportError:
            logging.getLogger(__name__).warning(
                "pywebview not installed, using browser. Install: pip install 'calibrie[native]'"
            )
            backend = "browser"
    task = GatingTask()
    return render(task, title="Calibrie Gating", channels_handler=_get_channels, backend=backend)


def cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Calibrie Gating UI")
    parser.add_argument("--browser", "-b", action="store_true", help="Open in browser instead of native window")
    args, _ = parser.parse_known_args()
    backend = "browser" if args.browser else "native"
    result = main(backend=backend)
    if result:
        print(f"Session: {len(result.files)} files, {len(result.gates)} gates")


if __name__ == "__main__":
    cli()
