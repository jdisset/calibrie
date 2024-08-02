## {{{                          --     import     --
from typing import (
    Annotated,
    List,
    Any,
    Optional,
)
from calibry.plots import CALIBRY_DEFAULT_DENSITY_CMAP, make_symlog_ax
import pandas as pd
import shapely.geometry as sg
import dracon as dr
import matplotlib.colors as mcolors
import dearpygui.dearpygui as dpg
from pathlib import Path
import xdialog
import calibry
import calibry.utils
from calibry.pipeline import Task
from calibry.utils import spline_biexponential, inverse_spline_biexponential
from pydantic import BaseModel, Field, ConfigDict
from numpy.typing import NDArray
import numpy as np
from calibry.calibrygui import Component, make_ui_field, unique_tag, ListManager, TextUI, DropdownUI
from matplotlib import pyplot as plt

##────────────────────────────────────────────────────────────────────────────}}}

AVAILABLE_AXIS = {'N/A'}


# -- UI ELEMENTS
## {{{                        --     plot tools     --
def powers_of_ten(xmin, xmax, skip10=False):
    # returns a list of powers of 10 that are between xmin and xmax,
    # as well as the list of subpoewrs of 10 that are between xmin and xmax (as a separate array)
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    powers = np.arange(logbounds[0], logbounds[1] + 1)
    p10 = np.power(10, np.abs(powers)) * np.sign(powers)
    subticks = np.concatenate([p * np.arange(1, 10) for p in p10 if p != 0])
    if skip10:
        p10 = np.delete(p10, np.where(np.abs(p10) == 10))
    return p10, subticks


# Set major ticks with improved labeling
def format_tick_label(p, skip10=True) -> str:
    if p == 0:
        return "0"
    elif np.abs(p) <= 100:
        if p == int(p):
            if np.abs(p) == 10 and skip10:
                return ""
            return str(int(p))
        return f"{p:.1f}"
    else:
        exponent = int(np.log10(np.abs(p)))
        sign = "+" if p > 0 else "-"
        return f"{sign}1e{exponent}"


COMPRESSION = 0.4
THRESHOLD = 200


def tr(x):
    return spline_biexponential(x, threshold=THRESHOLD, compression=COMPRESSION)


def invtr(y):
    return inverse_spline_biexponential(y, threshold=THRESHOLD, compression=COMPRESSION)


def make_dpg_symlog_ax(axis, lims, skip10=False):
    lims_tr = tr(np.array(lims))
    p10, subticks = powers_of_ten(*lims, skip10=skip10)
    major_ticks = tuple(zip([format_tick_label(p) for p in p10], tr(p10)))
    # a trick to get the major ticks to show up stronger since dpg doesn't support minor ticks:
    major_ticks_again = tuple(zip(['' for p in p10], tr(p10)))
    minor_ticks = tuple(zip(['' for p in subticks], tr(subticks)))
    dpg.set_axis_ticks(axis, minor_ticks + major_ticks + major_ticks_again * 2)

    return tr, invtr, lims_tr


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     polygondrawer     --


def rgb_float_to_int(rgb):
    return [int(c * 255) for c in rgb]


def random_color(hue_range=(0, 1), saturation_range=(0.4, 0.8), value_range=(0.9, 1)):
    h, s, v = [np.random.uniform(*r) for r in (hue_range, saturation_range, value_range)]
    color = mcolors.hsv_to_rgb((h, s, v))
    return [int(c * 255) for c in color] + [255]


class PolygonDrawer:
    def __init__(self):
        self.vertices = []
        self.adding_enabled = False
        self.delete_enabled = False
        self.color = random_color()
        self.plot = None
        self.y_axis = None
        self.controls_parent = None
        self.line_series_tag = None
        self.coord_transform_to_screen = tr
        self.inv_coord_transform = invtr
        self.on_update = None

    def setup_themes(self):
        with dpg.theme() as enabled_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (0, 120, 220, 255), category=dpg.mvThemeCat_Core
                )

        with dpg.theme() as disabled_theme:
            ...

        with dpg.theme() as line_series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots)
                self.line_color_tag = dpg.add_theme_color(
                    dpg.mvPlotCol_Line, self.color, category=dpg.mvThemeCat_Plots
                )

        self.enabled_theme = enabled_theme
        self.disabled_theme = disabled_theme
        self.line_series_theme = line_series_theme

    def add(self, parent_tag, plot_tag, y_axis):
        self.controls_parent = parent_tag
        self.plot = plot_tag
        self.y_axis = y_axis

        self.setup_themes()

        with dpg.group(parent=self.controls_parent, horizontal=True):
            self.add_btn = dpg.add_button(label="Add Points", callback=self.toggle_adding)
            self.del_btn = dpg.add_button(label="Del Points", callback=self.toggle_deleting)
            dpg.add_button(label="Delete All Points", callback=self.delete_all_points)
            self.color_picker = dpg.add_color_edit(
                default_value=self.color, no_inputs=True, no_alpha=False, callback=self.update_color
            )

        with dpg.item_handler_registry() as registry:
            dpg.add_item_clicked_handler(
                button=dpg.mvMouseButton_Right, callback=self.clicked_right
            )
        dpg.bind_item_handler_registry(self.plot, registry)

        self.line_series_tag = dpg.add_line_series([], [], parent=self.y_axis)
        self.apply_button_theme()

    def update_color(self, sender, app_data, user_data):
        self.color = [int(c * 255.0) for c in app_data]
        print("color:", self.color)
        dpg.set_value(self.line_color_tag, self.color)
        self.update_polygon()

    def apply_button_theme(self):
        if self.adding_enabled:
            dpg.bind_item_theme(self.add_btn, self.enabled_theme)
        else:
            dpg.bind_item_theme(self.add_btn, self.disabled_theme)

    def toggle_adding(self, sender, app_data, user_data):
        if self.adding_enabled:
            self.adding_enabled = False
        else:
            self.adding_enabled = True
            self.delete_enabled = False
        self.apply_button_theme()

    def toggle_deleting(self, sender, app_data, user_data):
        if self.delete_enabled:
            self.delete_enabled = False
        else:
            self.delete_enabled = True
            self.adding_enabled = False
        self.apply_button_theme()

    def find_closest_point(self, x, y):
        coords = self.get_point_coords(loop=False)
        if len(coords) == 0:
            return None
        dists = np.linalg.norm(coords[:, :2] - np.array([x, y]), axis=1)
        return np.argmin(dists)

    def find_closest_segment(self, x, y):
        if len(self.vertices) < 2:
            return None

        coords = self.get_point_coords(loop=True)
        best_segment = None
        best_dist = np.inf
        for i, segment in enumerate(zip(coords[:-1], coords[1:])):
            p1, p2 = segment
            d = sg.LineString([p1[:2], p2[:2]]).distance(sg.Point([x, y]))
            if d < best_dist:
                best_dist = d
                best_segment = i
        return best_segment

    def clicked_right(self, sender, app_data, user_data):
        if self.adding_enabled:
            self.add_point_click()
        elif self.delete_enabled:
            self.delete_point()

    def add_point_click(self):
        if not self.adding_enabled:
            return
        x, y = dpg.get_plot_mouse_pos()
        self.add_point(x, y)

    def add_point(self, x, y, screen_space=True):
        if not screen_space:
            x, y = self.coord_transform_to_screen(np.array([x, y]))

        assert self.plot is not None, "No plot, you need to call add() first"

        p = dpg.add_drag_point(
            default_value=(x, y),
            color=[255, 255, 255, 255],
            parent=self.plot,
            callback=self.update_point,
            show_label=False,
        )

        # let's find the closest point in the series:

        append_at = 0

        closest = self.find_closest_segment(x, y)
        if closest is not None:
            print("closest segment:", closest)
            append_at = closest + 1

        self.vertices.insert(append_at, p)
        self.update_polygon()

    def delete_point(self):
        if not self.delete_enabled:
            return
        x, y = dpg.get_plot_mouse_pos()
        closest = self.find_closest_point(x, y)
        if closest is not None:
            dpg.delete_item(self.vertices.pop(closest))
            self.update_polygon()

    def update_point(self, sender, app_data, user_data):
        self.update_polygon()

    def delete_all_points(self):
        for point in self.vertices:
            dpg.delete_item(point)
        self.vertices = []
        self.update_polygon()

    def get_point_coords(self, loop=False, transform=False):
        coords = [dpg.get_value(p) for p in self.vertices]
        if loop:
            coords.append(coords[0])
        coords = np.array(coords)
        if transform:
            coords = self.inv_coord_transform(coords)
        return coords

    def update_polygon(self):
        if len(self.vertices) > 2:
            coords = self.get_point_coords(loop=True).T.tolist()
            dpg.set_value(self.line_series_tag, coords)
        else:
            dpg.set_value(self.line_series_tag, [[], []])

        dpg.bind_item_theme(self.line_series_tag, self.line_series_theme)

        if self.on_update:
            self.on_update(self)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                         --     df series     --


def get_resampled(df, resample_to):
    total_points = len(df)
    resample_to = int(min(total_points, resample_to))
    resampled_df = df[:resample_to]
    return resampled_df


DEFAULT_RESAMPLE_TO = 50000
AX_MIN = -10000


class DataframeSerie(Component):
    dataframe: pd.DataFrame
    name: str
    resample_to: int = DEFAULT_RESAMPLE_TO
    color: List[int] = Field(default_factory=random_color)
    alpha: float = 0.1
    unique_id: Optional[str] = Field(default_factory=unique_tag)

    def __init__(self, **data):
        super().__init__(**data)

    def model_post_init(self, *a, **k):
        super().model_post_init(*a, **k)
        self._series = None
        self._shuffled_df = self.dataframe.sample(frac=1)
        self._parentplot = None

    def setup_theme(self):
        assert self._series is not None, "No _series, You need to call add() first"
        if len(self.color) == 3:
            self.color.append(255)
        self.color[3] = int(self.alpha * 255)
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_Marker,
                    dpg.mvPlotMarker_Cross,
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill, self.color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline, self.color, category=dpg.mvThemeCat_Plots
                )
        self._theme = theme
        dpg.bind_item_theme(self._series, self._theme)

    def get_min(self, axis):
        return self.dataframe[axis].min()

    def get_max(self, axis):
        return self.dataframe[axis].max()

    def resample(self, sender, app_data, user_data):
        self._resampled_df = get_resampled(self.dataframe, int(app_data))
        self.set_series_value()

    def set_series_value(self):
        assert self._parentplot is not None, "No _parentplot, You need to call add() first"
        assert self._series is not None, "You need to call add() first"
        xaxis, yaxis = self._parentplot.xaxis, self._parentplot.yaxis
        data = tr(self._resampled_df[[xaxis, yaxis]].values).T.tolist()
        dpg.set_value(self._series, data)

    def update_color(self, sender, app_data, user_data):
        self.color = [int(c * 255) for c in app_data]
        self.setup_theme()

    def update_alpha(self, sender, app_data, user_data):
        self.alpha = app_data
        self.setup_theme()

    def add(self, parent, **kwargs):
        assert self._parentplot is not None, "need a parent plot"
        self._series = dpg.add_scatter_series(
            [], [], label=self.name, parent=self._parentplot._y_axis
        )

        with dpg.group(parent=self._series):
            dpg.add_button(
                label="Remove", callback=self._parentplot.remove_series, user_data=self.unique_id
            )
            dpg.add_color_edit(
                default_value=self.color,
                no_inputs=True,
                no_alpha=True,
                callback=self.update_color,
            )
            dpg.add_slider_int(
                label="Resample to",
                min_value=0,
                max_value=len(self.dataframe),
                callback=self.resample,
                default_value=self.resample_to,
            )
            alphaslider = dpg.add_slider_float(
                label="Alpha",
                min_value=0,
                max_value=1,
                default_value=0.1,
                callback=self.update_alpha,
            )

        self.setup_theme()
        self._resampled_df = get_resampled(self.dataframe, self.resample_to)
        self.set_series_value()


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                         --     dplot     --


class DistributionPlot(Component):
    xaxis: str = ''
    yaxis: str = ''
    series: List[DataframeSerie] = []

    def model_post_init(self, *a, **k):
        super().model_post_init(*a, **k)
        for serie in self.series:
            serie._parentplot = self

    def remove_series(self, sender, app_data, user_data):
        assert len(self.series) > 0, "No series to remove"
        uid = user_data
        all_uids = [s.unique_id for s in self.series]
        print("all uids:", all_uids)
        print("uid:", uid)
        i = all_uids.index(uid)
        s = self.series.pop(i)
        dpg.delete_item(s._series)

    def update_series(self):
        if len(self.series) > 0:
            xmin = min([df.get_min(self.xaxis) for df in self.series])
            xmax = max([df.get_max(self.xaxis) for df in self.series])
            ymin = min([df.get_min(self.yaxis) for df in self.series])
            ymax = max([df.get_max(self.yaxis) for df in self.series])
            make_dpg_symlog_ax(self._x_axis, (min(xmin, AX_MIN), max(xmax * 10, xmax)))
            make_dpg_symlog_ax(self._y_axis, (min(ymin, AX_MIN), max(ymax * 10, ymax)))
            for serie in self.series:
                serie.set_series_value()

    def add_series(self, serie):
        self.series.append(serie)
        serie._parentplot = self
        serie.add(parent=self._plot)
        self.update_series()
        dpg.fit_axis_data(self._x_axis)
        dpg.fit_axis_data(self._y_axis)

    def add(self, parent, **_):
        self._plot = dpg.add_plot(
            label="Events",
            no_box_select=True,
            height=-20,
            width=-1,
            no_menus=True,
            equal_aspects=True,
            no_mouse_pos=True,
            fit_button=True,
            parent=parent,
        )
        self._legend = dpg.add_plot_legend(parent=self._plot)

        self._x_axis = dpg.add_plot_axis(
            dpg.mvXAxis, label=self.xaxis, no_gridlines=True, parent=self._plot
        )
        self._y_axis = dpg.add_plot_axis(
            dpg.mvYAxis, label=self.yaxis, no_gridlines=True, parent=self._plot
        )

        for serie in self.series:
            serie.add(parent=self._plot)

        # don't show grid lines:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvPlot):
                dpg.add_theme_color(
                    dpg.mvPlotCol_PlotBg, [0, 0, 0, 100], category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_XAxisGrid, [255, 255, 255, 50], category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_YAxisGrid, [255, 255, 255, 50], category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(self._plot, theme)
        self.update_series()


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                         --     datafile     --


def truncate(s, length, cut_at=0.3):
    if len(s) > length:
        cutpoint = int(length * cut_at)
        s = s[:cutpoint] + '...' + s[-(length - cutpoint - 3) :]
    return s


class DataFile(Component):
    path: Optional[str] = None
    df: pd.DataFrame = Field(default_factory=pd.DataFrame)

    def model_post_init(self, *args):
        super().model_post_init(*args)
        self.load()

    def load(self):
        df = calibry.utils.load_to_df(self.path)
        assert isinstance(df, pd.DataFrame), f"Expected a DataFrame, got {type(df)}"
        self.df = df
        AVAILABLE_AXIS.update(df.columns)

    def add(self, parent, **kwargs):
        path_short = Path(self.path).name if self.path else 'None'
        if 'tag' in kwargs:
            self._tag = kwargs['tag']
        # allow dragging the file to the plot
        dpg.add_text(
            f"{truncate(path_short,17)} ({len(self.df)} rows)",
            parent=parent,
            tag=self._tag,
            user_data=self,
            # payload_type='DataFile',
            # drag_callback=lambda: print('dragged'),
        )
        # dpg.add_drag_payload(parent=self.tag, drag_data="Payload Data")

    def _toggle_use(self, sender, app_data, user_data):
        self.use = app_data


def add_new_file(list_manager, *args, **kwargs):
    files = xdialog.open_file("Title Here", filetypes=[], multiple=True)
    loaded_files = [DataFile(path=f) for f in files]

    for file in loaded_files:
        list_manager.append(file)


class GatingFiles(Component):
    files: Annotated[
        List[DataFile],
        make_ui_field(
            ListManager,
            item_type=DataFile,
            add_button_callback=add_new_file,
            on_add_callback=lambda f: new_file_added(f),
            on_delete_callback=lambda f, _: file_deleted(f),
            button_label='Load File(s)',
        ),
    ] = []
    # Field(
        # default_factory=lambda: [
            # DataFile(
                # path='/Users/jeandisset/Dropbox (MIT)/Biocomp_v2/Experiments/2024-02-18_BPv4_BPv5/data/raw/2024-02-18_BPv4_BPv5_19.fcs'
            # )
        # ]
    # )


##────────────────────────────────────────────────────────────────────────────}}}

# -- TASK
## {{{                        --     PolygonGate     --


class PolygonGate(Component):
    vertices: List[List[float]] = []
    name: Annotated[str, make_ui_field(TextUI, width=150)] = ''
    xchannel: Annotated[
        str, make_ui_field(DropdownUI, get_available_items=lambda: list(AVAILABLE_AXIS), width=120)
    ]
    ychannel: Annotated[
        str, make_ui_field(DropdownUI, get_available_items=lambda: list(AVAILABLE_AXIS), width=100)
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

    def add_datafile(self, datafile):
        self._plot.add_series(DataframeSerie(dataframe=datafile.df, name=datafile.path))

    def delete_datafile(self, datafile):
        for serie in self._plot.series:
            if serie.dataframe is datafile.df:
                self._plot.remove_series(None, None, serie.unique_id)

    def __call__(self, df):
        if len(self.vertices) < 3:
            return df

        poly = sg.Polygon(np.asarray(self.vertices))

        def is_in_poly(row):
            return sg.Point(row[self.xchannel], row[self.ychannel]).within(poly)

        return df[df.apply(is_in_poly, axis=1)]

    def attr_changed(self, attr_name, old_value=None):
        print(f'attr {attr_name} changed from {old_value} to {getattr(self, attr_name)}')
        self.update_plot()

    def update_plot(self, *args, **kwargs):
        self._plot.xaxis = self.xchannel
        self._plot.yaxis = self.ychannel
        dpg.set_item_label(self._plot._x_axis, self.xchannel)
        dpg.set_item_label(self._plot._y_axis, self.ychannel)
        self._plot.update_series()
        # get coords from drawer and populate the vertices
        vertices = self._drawer.get_point_coords(transform=True)
        if len(vertices) > 0:
            vertices = vertices[:, :2]
        self.vertices = vertices.tolist()
        if dpg.does_item_exist(self._mainwin):
            dpg.set_item_label(self._mainwin, self.name)

    def add_window(self):
        # create a window for the plot + controls
        self._mainwin = dpg.add_window(label=self.name, width=700, height=700, pos=(100, 100))
        self._plot.add(parent=self._mainwin)
        self._drawer.add(
            parent_tag=self._mainwin, plot_tag=self._plot._plot, y_axis=self._plot._y_axis
        )

    def add(self, parent, **kwargs):
        self._drawer: PolygonDrawer = PolygonDrawer()
        self._plot: DistributionPlot = DistributionPlot()
        self._drawer.adding_enabled = True
        if len(self.vertices) > 0:
            # we populate the drawer with the vertices
            self._drawer.vertices.clear()
            for x, y in self.vertices:
                self._drawer.add_point(x, y, screen_space=False)
        self._drawer.on_update = self.update_plot

        self.add_window()
        self._ui_grp_kwargs = {'horizontal': True}
        tag = super().add(parent=self._mainwin, **kwargs)  # add the name, xchannel, ychannel fields
        # move it to _mainwin, because for some reason it's not working when added to parent
        self._btn = dpg.add_button(label='focus', callback=self.focus, parent=parent)
        self.update_plot()
        return tag

    def focus(self):
        if dpg.does_item_exist(self._mainwin):
            dpg.focus_item(self._mainwin)
            dpg.show_item(self._mainwin)
        else:
            self.add_window()

    def delete(self):
        super().delete()
        if dpg.does_item_exist(self._btn):
            dpg.delete_item(self._btn)
        if dpg.does_item_exist(self._mainwin):
            dpg.delete_item(self._mainwin)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     plot utils     --


def density_histogram2d(
    ax,
    X,
    Y,
    xrange,
    yrange,
    nbins,
    cmap=None,
    vmin: Optional[float] = 0.01,
    vmax: Optional[float] = 1,
    noise_smooth=0,
):
    if isinstance(nbins, int):
        nbins = (nbins, nbins)

    xres = np.abs(np.subtract(*xrange)) / nbins[0]
    yres = np.abs(np.subtract(*yrange)) / nbins[1]

    X = X + np.random.normal(size=X.shape) * noise_smooth * xres
    Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=nbins,
        density=False,
        range=[xrange, yrange],
    )

    h = np.log10(h + 1).T
    if cmap is None:
        cmap = CALIBRY_DEFAULT_DENSITY_CMAP

    ax.imshow(
        h,
        extent=[*xrange, *yrange],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     GatingTask     --
class GatingTask(Task, Component):
    gates: Annotated[
        List[PolygonGate],
        make_ui_field(
            ListManager,
            item_type=PolygonGate,
            on_add_callback=lambda g: new_gate_added(g),
        ),
    ] = []

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    def initialize(self, ctx: Any) -> Any:
        return {'cell_data_loader': self.load_cells}

    def load_cells(self, data, column_order=None) -> Any:
        # first load with all columns, and apply the gates
        df = calibry.utils.load_to_df(data)
        df = self.apply_all_gates(df)
        if column_order is not None:
            return df[column_order]
        return df

    def apply_all_gates(self, df):
        for gate in self.gates:
            df = gate(df)
        return df

    def process(self, ctx: Any) -> Any: ...

    def diagnostics(
        self,
        ctx: Any,
        use_files: list,
        resample_to=250000,
        nbins=200,
        density_lims=(1e-6, None),
        **_,
    ) -> Any:
        assert use_files, "No files selected"

        all_data_df = pd.concat([calibry.utils.load_to_df(f) for f in use_files])

        all_columns = all_data_df.columns
        print("all columns:", all_columns)

        df = get_resampled(all_data_df, resample_to)

        # create a plot for each gates groupd by the same x and y axis
        gates_per_axis_pair = {}

        for gate in self.gates:
            if len(gate.vertices) > 2:
                # sort x and y axis to group the gates
                key = tuple(sorted([gate.xchannel, gate.ychannel]))
                if key not in gates_per_axis_pair:
                    gates_per_axis_pair[key] = []
                gates_per_axis_pair[key].append(gate)

        n_figures = len(gates_per_axis_pair)
        # compute best layout:
        best_n_cols = int(np.ceil(np.sqrt(n_figures)))
        n_cols = min(best_n_cols, 3)
        n_rows = int(np.ceil(n_figures / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows), dpi=200)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        flat_axes = axes.flatten()

        for i, ((xaxis, yaxis), gates) in enumerate(gates_per_axis_pair.items()):
            ax = flat_axes[i]
            print(f"Plotting {xaxis} vs {yaxis}")
            X = df[[xaxis, yaxis]].values
            xlims = np.min(X[:, 0]), np.max(X[:, 0])
            ylims = np.min(X[:, 1]), np.max(X[:, 1])
            tr, invtr, xlims_tr, ylims_tr = make_symlog_ax(ax, xlims, ylims)
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
            # show the gates:
            for _, gate in enumerate(gates):
                if len(gate.vertices) > 2:
                    gcol = random_color()
                    gcol = [c / 255 for c in gcol]
                    vertices = np.array(gate.vertices)
                    # make it loop:
                    if len(vertices) > 0:
                        vertices = np.vstack([vertices, vertices[0]])
                    vertices = tr(vertices)
                    if gate.xchannel == xaxis:
                        x, y = vertices.T
                    else:
                        y, x = vertices.T
                    ax.plot(x, y, color=gcol, lw=3)
                    ax.fill(x, y, color=gcol, alpha=0.2)
                    rightmost = vertices[np.argmax(vertices[:, 0])]
                    after_applying = gate(df)
                    percent = (len(after_applying) / len(df)) * 100
                    stat_text = f"{gate.name}: {len(after_applying)}/{len(df)} ({percent:.1f}\%)"

                    ax.text(
                        rightmost[0],
                        rightmost[1],
                        stat_text,
                        color='black',
                        weight='bold',
                        backgroundcolor=gcol,
                        ha='left',
                        va='center',
                    )


            ax.axvline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')
            ax.axhline(0, c='k', lw=0.5, alpha=0.5, linestyle='--')

            ax.set_aspect('equal')
            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)

        after_all_gates = self.apply_all_gates(df)
        percent = (len(after_all_gates) / len(df)) * 100
        stat_text = f"Total events kept: {len(after_all_gates)}/{len(df)} ({percent:.1f}\%)"

        fig.text(0.5, 0.01, stat_text, ha='center', va='center', fontsize=12)

        fig.tight_layout()
        return fig


##────────────────────────────────────────────────────────────────────────────}}}
