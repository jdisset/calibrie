import xdialog
import dearpygui.dearpygui as dpg
import dracon as dr
from calibry.gating import GatingTask, GatingFiles
import numpy as np
import pandas as pd

START_WIDTH = 1100
START_HEIGHT = 900
LEFT_PANEL_WIDTH = 340
N = int(1e5)
dfs = [
    pd.DataFrame({'x': np.random.normal(0, 1000, N), 'y': np.random.normal(0, 1000, N)})
    for _ in range(3)
]

dpg.create_context()
dpg.create_viewport(title='Calibry Gating', width=START_WIDTH, height=START_HEIGHT)
dpg.setup_dearpygui()


gating_task = GatingTask()
gating_files = GatingFiles()


def new_file_added(file):
    global gating_task
    for gate in gating_task.gates:
        gate.add_datafile(file)


def file_deleted(file):
    global gating_task
    for gate in gating_task.gates:
        gate.delete_datafile(file)


def new_gate_added(gate):
    global gating_files
    for file in gating_files.files:
        gate.add_datafile(file)


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
    file = xdialog.open_file("Open gating task", filetypes=[('gating_task', '*.yaml')])
    global gating_task
    new_gating_task = load_gating_task(file)
    print(f'Loaded gating task: {new_gating_task}')
    for gate in new_gating_task.gates:
        gate.delete()
    gating_task.delete()
    gating_task = new_gating_task
    gating_task.add('left_panel', indent=5)


def detailed_diagnostics():
    fig = gating_task.diagnostics(None, use_files=gating_files.files)
    # save to in memory png:
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    png = buf.read()
    buf.close()

    # create window if it doesn't exist
    if not dpg.does_item_exist('detailed_diagnostics'):
        dpg.add_window(
            tag='detailed_diagnostics', width=800, height=800, label='Detailed Diagnostics'
        )
        # show the image in it
        dpg.add_image(
            'detailed_diagnostics_image',
            value=fig,
            width=800,
            height=800,
            parent='detailed_diagnostics',
        )
    else:
        dpg.set_value('detailed_diagnostics_image', png)
        dpg.show_item('detailed_diagnostics')
        dpg.focus_item('detailed_diagnostics')


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
        dpg.add_text("Gates")
        dpg.add_separator()
        gating_task.add('left_panel', indent=5)
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_text("Data")
        gating_files.add('left_panel')
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
