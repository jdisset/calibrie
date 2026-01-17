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
from declair_ui.layout import Cols, Panel, Rows, Sidebar, UIConfig
from declair_ui.widgets import ColorPicker, DensityPlot, Dropdown, FileList, PolygonDrawer, TreeList

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
) -> list[bool]:
    if len(vertices) < 3:
        return [False] * len(x)
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    xi, yi, xj, yj = v[:, 0], v[:, 1], np.roll(v[:, 0], -1), np.roll(v[:, 1], -1)
    dy = yj - yi
    cond = (yi[:, None] > y_arr) ^ (yj[:, None] > y_arr)
    t = np.where(
        dy[:, None] != 0, (y_arr - yi[:, None]) / dy[:, None] * (xj - xi)[:, None] + xi[:, None], 0
    )
    return ((np.count_nonzero(cond & (x_arr < t), axis=0) & 1).astype(bool)).tolist()


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
    gate_mask = np.array(points_in_polygon(vertices, x, y))
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
    channels: list[str] = Field(default_factory=list)
    preview_data: list[list[float]] | None = Field(default=None)
    preview_data_packed: dict | None = Field(default=None)

    @computed_field
    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1] if "/" in self.path else self.path


@client_runnable(helpers=[escape_name], packages=["numpy", "pydantic"])
def load_fcs_preview(file_bytes: bytes, filename: str, report_progress=None) -> DataFile:
    import fcsparser
    import numpy as np
    import base64

    if report_progress:
        report_progress(0.1, "Parsing header...")
    with open(f"/tmp/{filename}", "wb") as f:
        f.write(file_bytes)
    if report_progress:
        report_progress(0.3, "Decoding events...")
    _, data = fcsparser.parse(f"/tmp/{filename}", reformat_meta=True)
    if report_progress:
        report_progress(0.7, "Packing preview...")
    arr = data.to_numpy(dtype=np.float32, copy=False)
    packed = {
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }
    if report_progress:
        report_progress(0.95, "Finalizing...")
    return DataFile(
        path=filename,
        point_count=len(data),
        channels=[escape_name(c) for c in data.columns],
        preview_data=None,
        preview_data_packed=packed,
        visible=True,
    )


load_fcs_preview.supports_progress = True


@client_runnable
def compute_plot_data(files: list[dict], xaxis: str, yaxis: str) -> dict:
    """Compute multi-source plot data client-side from cached file data."""
    import numpy as np

    if not xaxis or not yaxis:
        return {}
    sources = []
    for f in files:
        if not f.get("visible", True) or not f.get("preview_data") or not f.get("channels"):
            continue
        channels = f["channels"]
        if xaxis not in channels or yaxis not in channels:
            continue
        xi, yi = channels.index(xaxis), channels.index(yaxis)
        data = f["preview_data"]
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] <= max(xi, yi):
            continue
        # JS pre-samples to <= 1M points when possible; keep as fallback.
        x = arr[:, xi].tolist()
        y = arr[:, yi].tolist()
        name = f.get("path", "").rsplit("/", 1)[-1]
        name = name.rsplit(".", 1)[0] if "." in name else name
        sources.append({"x": x, "y": y, "label": name})
    return {"sources": sources} if sources else {}


class PolygonGate(BaseModel):
    name: str = "Gate"
    xchannel: Chan = ""
    ychannel: Chan = ""
    vertices: Annotated[list[tuple[float, float]], UI(PolygonDrawer(), uses="main-plot")] = Field(
        default_factory=list
    )
    color: Annotated[tuple[int, int, int, int], UI(ColorPicker())] = (100, 200, 255, 200)
    applied: bool = False
    parent_gate: str | None = None
    event_count: int = Field(default=0, exclude=True)
    percent_of_parent: float = Field(default=0.0, exclude=True)
    stats_summary: str = Field(default="No data", exclude=True)

    def contains_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array(points_in_polygon(list(self.vertices), x.tolist(), y.tolist()), dtype=bool)

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
        list[DataFile], UI(FileList(file_processor=load_fcs_preview, badge_field="point_count", cache_fields=["preview_data"]))
    ] = Field(default_factory=list, exclude=True)
    xaxis: Annotated[Chan, UI(label="X Channel")] = Field(default="", exclude=True)
    yaxis: Annotated[Chan, UI(label="Y Channel")] = Field(default="", exclude=True)
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
            ),
            provides="main-plot",
        ),
        ClientComputed(compute=compute_plot_data, depends_on=["files", "xaxis", "yaxis"]),
    ] = Field(default_factory=dict, exclude=True)
    selected_gate_index: int = Field(default=-1, exclude=True)

    ui_config: ClassVar[UIConfig] = UIConfig(
        layout=Cols(
            Sidebar("files", "gates", width=300),
            Panel("plot_data", overlays=[("gates", "vertices")]),
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


def main() -> GatingTask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    task = GatingTask()
    return render(task, title="Calibrie Gating", channels_handler=_get_channels)


def cli() -> None:
    if len(sys.argv) > 1:
        logging.getLogger(__name__).info("CLI args ignored. Use UI file picker.")
    result = main()
    if result:
        print(f"Session: {len(result.files)} files, {len(result.gates)} gates")


if __name__ == "__main__":
    cli()
