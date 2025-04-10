# calibrie/gui/main.py

import dearpygui.dearpygui as dpg
from calibrie.gui.core import TaskRegistry  # keep this import
from calibrie.gui.pipeline_manager import PipelineManager
# no need to import calibrie here unless for other reasons


def start_gui():
    """Initializes and starts the Calibrie GUI."""

    START_WIDTH = 1400
    START_HEIGHT = 900

    dpg.create_context()
    TaskRegistry.discover_tasks()  # <<<--- add discovery call here

    dpg.configure_app(manual_callback_management=True)

    pipeline_manager = PipelineManager()

    # --- Callbacks ---
    def _add_task_callback(sender, app_data, user_data):
        task_name = user_data
        task_class = TaskRegistry.get_task_class(task_name)
        if task_class:
            pipeline_manager.add_task(task_class)
        else:
            print(f"Error: Task class '{task_name}' not found in registry.")

    def _run_init_callback():
        pipeline_manager.initialize_pipeline()

    def _run_all_diag_callback():
        pipeline_manager.run_all_diagnostics()

    def _apply_pipeline_callback():
        pipeline_manager.apply_pipeline_to_file()

    def _save_callback():
        pipeline_manager.save_pipeline()

    def _load_callback():
        pipeline_manager.load_pipeline()

    # --- Main Window ---
    main_window_tag = "MainWindow"  # give the main window an explicit tag
    with dpg.window(tag=main_window_tag, label="Calibrie Pipeline Editor"):
        # --- Menu Bar ---
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Load Pipeline...", callback=_load_callback)
                dpg.add_menu_item(label="Save Pipeline As...", callback=_save_callback)
                # add quit later

            with dpg.menu(label="Pipeline"):
                dpg.add_menu_item(label="Initialize Pipeline", callback=_run_init_callback)
                dpg.add_menu_item(label="Run All Diagnostics", callback=_run_all_diag_callback)
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Apply Pipeline to Experiment...", callback=_apply_pipeline_callback
                )

            with dpg.menu(label="Tasks"):
                # populate dynamically from registry
                # now the registry should be populated before the menu is created
                registered_tasks = TaskRegistry.get_registered_tasks()
                if not registered_tasks:
                    dpg.add_menu_item(label="No tasks found/registered", enabled=False)
                else:
                    with dpg.menu(label="Add Task"):
                        for task_name in sorted(registered_tasks.keys()):
                            dpg.add_menu_item(
                                label=task_name, callback=_add_task_callback, user_data=task_name
                            )

        # --- Main Content Area ---
        pipeline_manager.add(parent=main_window_tag)

    dpg.create_viewport(title='Calibrie GUI', width=START_WIDTH, height=START_HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(main_window_tag, True)  # use the tag

    # --- Main Loop ---
    while dpg.is_dearpygui_running():
        jobs = dpg.get_callback_queue()
        dpg.run_callbacks(jobs)
        dpg.render_dearpygui_frame()

    # --- Cleanup ---
    print("Cleaning up GUI...")
    pipeline_manager.delete()
    print("Destroying context...")
    dpg.destroy_context()
    print("GUI Closed.")


if __name__ == '__main__':
    start_gui()
