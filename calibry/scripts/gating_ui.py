import xdialog
import dearpygui.dearpygui as dpg
import dracon as dr
from calibry.gating import GatingTask, GatingFiles, PolygonGate
import numpy as np
import pandas as pd

START_WIDTH = 1300
START_HEIGHT = 1000
LEFT_PANEL_WIDTH = 340


def export():
    file = xdialog.save_file("Save gating task", filetypes=[('gating_task', '*.yaml')])
    loader = dr.DraconLoader()
    loader.yaml.representer.full_module_path = False
    dmp = loader.dump(gating_task)
    print(dmp)
    with open(file, 'w') as f:
        f.write(dmp)


def load_gating_task(file):
    with open(file, 'r') as f:
        gt_str = f.read()
    loader = dr.DraconLoader()
    gating_task = loader.loads(gt_str)
    return gating_task


def load():
    file = xdialog.open_file("Open gating task")
    global gating_task, gating_files
    new_gating_task = load_gating_task(file)
    print(f'Loaded gating task: {new_gating_task}')
    for gate in new_gating_task.gates:
        gate.delete()
    new_gating_task._gating_files = gating_files
    gating_files._gating_task = new_gating_task
    gating_task.delete()
    gating_task = new_gating_task
    gating_task.add('left_panel', indent=5)


def detailed_diagnostics():
    # use a image backend
    from io import BytesIO
    import matplotlib

    matplotlib.use('Agg')
    WINSIZE = 1000
    IMSIZE = 3000
    dpi = 400
    empty_image = np.zeros((IMSIZE, IMSIZE, 4)).flatten().tolist()

    # create window if it doesn't exist
    if not dpg.does_item_exist('detailed_diagnostics'):
        dpg.add_window(
            tag='detailed_diagnostics', width=WINSIZE, height=WINSIZE, label='Detailed Diagnostics'
        )
        if not dpg.does_item_exist('loading-detailed-diagnostics'):
            dpg.add_loading_indicator(
                parent='detailed_diagnostics', tag='loading-detailed-diagnostics'
            )
        # show the image in it
        # create the texture:
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=IMSIZE,
                height=IMSIZE,
                tag="detailed_diagnostics_texture",
                default_value=empty_image,
            )

        with dpg.plot(
            label="Image Plot",
            height=-1,
            width=-1,
            parent='detailed_diagnostics',
            no_menus=True,
            no_mouse_pos=True,
            equal_aspects=True,
        ) as dpg_plot:
            dpg.add_plot_legend()
            dpg.add_plot_axis(
                dpg.mvXAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True, tag='diag_xaxis'
            )
            with dpg.plot_axis(
                dpg.mvYAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True, tag='diag_yaxis'
            ) as yax:
                dpg.add_image_series(
                    "detailed_diagnostics_texture", [0, 0], [IMSIZE, IMSIZE], label=""
                )
    else:
        dpg.show_item('detailed_diagnostics')
        dpg.focus_item('detailed_diagnostics')

    if not gating_files.files:
        with dpg.window(modal=True, autosize=True) as p:
            dpg.add_text('No files selected', parent=p)
        return

    dpg.show_item('loading-detailed-diagnostics')

    dfig = gating_task.diagnostics(
        None,
        use_files=[f.path for f in gating_files.files],
        figsize=(IMSIZE / dpi, IMSIZE / dpi),
        dpi=dpi,
    )

    fig = dfig[0].fig
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    assert (
        image_from_plot.shape[0] == IMSIZE * IMSIZE * 3
    ), f'Expected {IMSIZE * IMSIZE * 3}, got {image_from_plot.shape[0]}'
    # it's flat, which is fine, but we need to add the alpha channel (every 4th element)
    image_rgba = image_from_plot.reshape((IMSIZE, IMSIZE, 3)).astype(np.float32) / 255
    image_rgba = np.concatenate((image_rgba, np.ones((IMSIZE, IMSIZE, 1))), axis=2)
    image_as_float = image_rgba.flatten().tolist()
    dpg.set_value('detailed_diagnostics_texture', image_as_float)
    dpg.hide_item('loading-detailed-diagnostics')
    dpg.fit_axis_data('diag_xaxis')
    dpg.fit_axis_data('diag_yaxis')


def main():
    global gating_task, gating_files

    dpg.create_context()
    dpg.create_viewport(title='Calibry Gating', width=START_WIDTH, height=START_HEIGHT)
    dpg.setup_dearpygui()
    gating_task = GatingTask()
    gating_files = GatingFiles()

    gating_task._gating_files = gating_files
    gating_files._gating_task = gating_task

    with dpg.window(tag='calibrygating', pos=(0, 0)):
        # Theme setting
        dpg.set_primary_window('calibrygating', True)
        with dpg.child_window(
            label="parameters",
            width=LEFT_PANEL_WIDTH,
            height=-1,
            pos=[0, 0],
            tag='left_panel',
        ):
            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_text("Data")
            gating_files.add('left_panel')

            dpg.add_spacer(height=8)
            dpg.add_text("Gates")
            dpg.add_separator()
            gating_task.add('left_panel', indent=5)

            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_button(label='Generate Detailed Diagnostics', callback=detailed_diagnostics)

        # add a menu with an export button
        with dpg.menu_bar():
            with dpg.menu(label='File'):
                dpg.add_menu_item(label='Load', callback=load)
                dpg.add_menu_item(label='Export to yaml', callback=export)

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    print(f'Gating task: {gating_task}')
