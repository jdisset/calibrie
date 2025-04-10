import dearpygui.dearpygui as dpg
from calibrie.gui.core import TaskRegistry, Component, unique_tag
from calibrie.gui.task_view import TaskView
from calibrie.pipeline import Pipeline, Task
from calibrie.utils import Context
from typing import Dict, List, Optional, Type, Any
import dracon as dr
import xdialog
import yaml
from pathlib import Path


class PipelineManager(Component):
    """Manages the pipeline state and its representation in the GUI."""

    pipeline: Pipeline = Pipeline(tasks={})
    context: Context = Context()
    task_views: Dict[str, TaskView] = {}  # map task_name to TaskView instance
    ordered_task_names: List[str] = []

    _tab_bar_tag: str = ""
    _next_task_id: Dict[str, int] = {}  # e.g. {'LoadControls': 1, 'LinearCompensation': 2}

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.pipeline = Pipeline(tasks={})
        self.context = Context()
        self.task_views = {}
        self.ordered_task_names = []
        self._tab_bar_tag = unique_tag()
        self._next_task_id = {}

    def add(self, parent: str | int, **kwargs) -> str | int:
        """Adds the main tab bar for managing tasks."""
        # print(f"Adding PipelineManager UI (Tab Bar: {self._tab_bar_tag}) to parent {parent}")
        # The manager itself doesn't have much UI other than the tab bar
        self._ui_tag = dpg.add_tab_bar(
            parent=parent,
            tag=self._tab_bar_tag,
            reorderable=True,
            callback=self._handle_tab_reorder,
        )
        # Add initial tasks if any (e.g., when loading a pipeline)
        self._rebuild_tabs()
        return self._ui_tag

    def add_task(self, task_class: Type[Task]):
        """Adds a new task to the pipeline and the UI."""
        task_base_name = task_class.__name__
        # Generate unique name (e.g., LoadControls_1, LoadControls_2)
        instance_id = self._next_task_id.get(task_base_name, 0) + 1
        self._next_task_id[task_base_name] = instance_id
        task_name = f"{task_base_name}_{instance_id}"

        # print(f"Adding task: {task_name} (Class: {task_base_name})")

        # Create task instance with default values
        try:
            new_task = task_class()
            new_task._name = task_name  # Assign the generated name
        except Exception as e:
            print(f"Error instantiating task {task_base_name}: {e}")
            # Show error popup?
            return

        # Add to pipeline object
        self.pipeline.tasks[task_name] = new_task
        self.ordered_task_names.append(task_name)

        # Add to UI
        self._add_task_tab(task_name, new_task)
        # print(f"Task {task_name} added. Current order: {self.ordered_task_names}")

    def _add_task_tab(self, task_name: str, task_instance: Task):
        """Adds a DPG tab for a given task."""
        if not dpg.does_item_exist(self._tab_bar_tag):
            print(
                f"Error: Tab bar '{self._tab_bar_tag}' does not exist when trying to add tab for '{task_name}'."
            )
            return

        # optional check: is it actually a tab bar?
        # if dpg.get_item_info(self._tab_bar_tag)['type'] != 'mvAppItemType::mvTabBar':
        #     print(f"Error: Item '{self._tab_bar_tag}' is not a Tab Bar.")
        #     return

        tab_tag = f"tab_{task_name}"
        view_tag = f"view_{task_name}"
        tab_label = f"{self.ordered_task_names.index(task_name) + 1} - {task_name}"

        with dpg.tab(label=tab_label, parent=self._tab_bar_tag, tag=tab_tag, user_data=task_name):
            # add context menu for removing task
            with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Right):
                dpg.add_text(f"Task: {task_name}")
                dpg.add_separator()
                dpg.add_menu_item(label="Remove Task", callback=lambda: self.remove_task(task_name))
                # add move up/down later if drag-reorder fails

            task_view = TaskView()
            # ensure the view is parented to the tab
            task_view.add(
                dpg.last_item(), task_instance=task_instance, pipeline_manager=self, tag=view_tag
            )
            self.task_views[task_name] = task_view

    def remove_task(self, task_name: str):
        """Removes a task from the pipeline and UI."""
        if task_name not in self.pipeline.tasks:
            print(f"Error: Task '{task_name}' not found.")
            return

        print(f"Removing task: {task_name}")

        # Remove from pipeline object
        del self.pipeline.tasks[task_name]
        self.ordered_task_names.remove(task_name)

        # Remove from UI
        if task_name in self.task_views:
            self.task_views[task_name].delete()
            del self.task_views[task_name]

        # Delete the tab itself
        tab_tag = f"tab_{task_name}"
        if dpg.does_item_exist(tab_tag):
            dpg.delete_item(tab_tag)

        self._update_tab_labels()
        print(f"Task {task_name} removed. Current order: {self.ordered_task_names}")

    def _handle_tab_reorder(self, sender, app_data):
        """Callback when tabs are reordered by the user."""
        if not dpg.is_item_enabled(sender):
            return  # prevent callback during setup

        # print("Tab reorder detected.")
        new_order_tags = dpg.get_item_children(sender, 1)  # children slot 1 are the tabs
        new_ordered_task_names = [
            dpg.get_item_user_data(tag)
            for tag in new_order_tags
            if dpg.get_item_user_data(tag) is not None
        ]

        if new_ordered_task_names != self.ordered_task_names:
            # print(f"  Old order: {self.ordered_task_names}")
            # print(f"  New order: {new_ordered_task_names}")
            self.ordered_task_names = new_ordered_task_names
            # Update the underlying pipeline task order (important!)
            ordered_tasks = {name: self.pipeline.tasks[name] for name in self.ordered_task_names}
            self.pipeline.tasks = ordered_tasks
            self._update_tab_labels()
            print("Pipeline task order updated.")
        # else:
        # print("  Order hasn't changed.")

    def _update_tab_labels(self):
        """Updates the numbers in the tab labels based on the current order."""
        for i, task_name in enumerate(self.ordered_task_names):
            tab_tag = f"tab_{task_name}"
            if dpg.does_item_exist(tab_tag):
                new_label = f"{i + 1} - {task_name}"
                dpg.set_item_label(tab_tag, new_label)

    def _rebuild_tabs(self):
        """Clears and rebuilds all tabs based on the current pipeline state."""
        # print("Rebuilding tabs...")
        # Disable reorder callback during rebuild
        if dpg.does_item_exist(self._tab_bar_tag):
            dpg.configure_item(self._tab_bar_tag, callback=None)

        # Clear existing views and tabs
        for view in self.task_views.values():
            view.delete()
        self.task_views.clear()
        if dpg.does_item_exist(self._tab_bar_tag):
            # Delete existing tabs safely
            tabs_to_delete = dpg.get_item_children(self._tab_bar_tag, 1)
            # print(f" Deleting {len(tabs_to_delete)} existing tabs...")
            for tab_tag in tabs_to_delete:
                if dpg.does_item_exist(tab_tag):
                    dpg.delete_item(tab_tag)

        # Re-add tabs in the correct order
        # print(f" Re-adding {len(self.ordered_task_names)} tabs...")
        for task_name in self.ordered_task_names:
            task_instance = self.pipeline.tasks.get(task_name)
            if task_instance:
                self._add_task_tab(task_name, task_instance)
            else:
                print(f"Warning: Task instance for '{task_name}' not found during rebuild.")

        # Re-enable reorder callback
        if dpg.does_item_exist(self._tab_bar_tag):
            dpg.configure_item(self._tab_bar_tag, callback=self._handle_tab_reorder)

    def initialize_pipeline(self):
        """Initializes the entire pipeline."""
        if not self.pipeline.tasks:
            print("Pipeline is empty. Nothing to initialize.")
            return
        print("Initializing pipeline...")
        try:
            # Ensure tasks have names and are ordered correctly in the pipeline object
            self.pipeline.tasks = {
                name: self.pipeline.tasks[name] for name in self.ordered_task_names
            }
            for name in self.ordered_task_names:
                self.pipeline.tasks[name]._name = name  # Ensure name is set

            self.pipeline.order_tasks_and_assign_names()  # Set internal ordered list

            # Reset context before full initialization
            self.context = Context()
            self.pipeline.initialize()  # This internally updates the pipeline's context
            self.context = self.pipeline._context  # Get the context from the pipeline instance

            print("Pipeline initialized successfully.")
            print("Running all diagnostics after initialization...")
            self.run_all_diagnostics()

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            # Show error popup?
            import traceback

            traceback.print_exc()

    def run_all_diagnostics(self):
        """Runs diagnostics for all tasks in the pipeline."""
        if not self.context:  # Check if context exists (pipeline initialized)
            print("Pipeline not initialized. Cannot run diagnostics.")
            # Optionally prompt user to initialize?
            return
        print("Running all diagnostics...")
        try:
            all_figures = self.pipeline.all_diagnostics()
            print(f"Generated {len(all_figures)} total figures.")
            # Distribute figures to the correct TaskView panels
            figs_by_task = {}
            for fig in all_figures:
                task_name = fig.source_task
                # Handle potential prefix like "01_"
                if task_name and '_' in task_name and task_name.split('_')[0].isdigit():
                    task_name = '_'.join(task_name.split('_')[1:])

                if task_name not in figs_by_task:
                    figs_by_task[task_name] = []
                figs_by_task[task_name].append(fig)

            for task_name, view in self.task_views.items():
                view.diagnostics_panel.set_figures(figs_by_task.get(task_name, []))

        except Exception as e:
            print(f"Error running all diagnostics: {e}")
            import traceback

            traceback.print_exc()

    def apply_pipeline_to_file(self):
        """Applies the configured pipeline to an experiment file."""
        if not self.context:  # Check if context exists (pipeline initialized)
            print("Pipeline not initialized. Cannot apply.")
            # Optionally prompt user to initialize?
            return

        # In a real scenario, this would load the experiment json5 file
        # and likely call the logic from calibrie-run script.
        # For now, just a placeholder.
        print("Placeholder: Applying pipeline to file...")
        exp_file = xdialog.open_file(
            "Select Experiment JSON file", filetypes=[("JSON", "*.json"), ("JSON5", "*.json5")]
        )
        if not exp_file:
            print("Application cancelled.")
            return

        print(f"Applying pipeline to {exp_file}...")
        print("NOTE: This functionality is not fully implemented in the GUI yet.")
        print("      It would typically involve parsing the experiment file")
        print("      and iterating calibrie.pipeline.apply_all() over samples.")
        # Add placeholder logic or integrate with run_pipeline.py's logic later.

    def load_pipeline(self):
        """Loads a pipeline from a YAML file."""
        file = xdialog.open_file("Load Pipeline", filetypes=[('YAML', '*.yaml')])
        if not file:
            return

        print(f"Loading pipeline from {file}...")
        try:
            with open(file, 'r') as f:
                pipeline_yaml = f.read()

            loader = dr.DraconLoader(context=TaskRegistry.get_registered_tasks())
            loaded_pipeline = loader.loads(pipeline_yaml)

            if not isinstance(loaded_pipeline, Pipeline):
                raise TypeError("Loaded file is not a valid Calibrie Pipeline.")

            self.pipeline = loaded_pipeline
            # Reconstruct internal state from loaded pipeline
            self.ordered_task_names = list(self.pipeline.tasks.keys())
            self._next_task_id = {}  # Reset instance IDs
            for name in self.ordered_task_names:  # Recalculate next IDs
                base_name = name.split('_')[0]
                instance_id = (
                    int(name.split('_')[-1]) if '_' in name and name.split('_')[-1].isdigit() else 0
                )
                self._next_task_id[base_name] = max(
                    self._next_task_id.get(base_name, 0), instance_id
                )
                self.pipeline.tasks[name]._name = name  # Ensure name set

            self.context = Context()  # Reset context
            self._rebuild_tabs()  # Update UI
            print("Pipeline loaded successfully.")

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            import traceback

            traceback.print_exc()
            # Show error popup?

    def save_pipeline(self):
        """Saves the current pipeline to a YAML file."""
        if not self.pipeline.tasks:
            print("Pipeline is empty. Nothing to save.")
            return

        file = xdialog.save_file("Save Pipeline", filetypes=[('YAML', '*.yaml')])
        if not file:
            return
        if not file.lower().endswith('.yaml'):
            file += '.yaml'

        print(f"Saving pipeline to {file}...")
        try:
            # Ensure tasks are in the correct order for saving
            self.pipeline.tasks = {
                name: self.pipeline.tasks[name] for name in self.ordered_task_names
            }

            # Use dracon dump
            pipeline_dump = dr.dump(self.pipeline)
            with open(file, 'w') as f:
                f.write(pipeline_dump)
            print("Pipeline saved successfully.")
        except Exception as e:
            print(f"Error saving pipeline: {e}")
            import traceback

            traceback.print_exc()
            # Show error popup?

    def get_context_for_task(self, task_name: str) -> Context:
        """
        Provides the context *as it would be* just before the specified task runs
        during a full pipeline initialization. This is complex to do perfectly
        without re-running part of the pipeline.
        For now, returns the current full context. Task initialize methods need
        to be robust to potentially having future task data present.
        """
        # TODO: Implement more accurately if needed, possibly by caching context states.
        return self.context.copy(deep=False)  # Return a shallow copy

    def update_context(self, new_context_data: Context):
        """Updates the manager's context with data from a task."""
        self.context.update(new_context_data)

    def delete(self):
        """Cleans up the manager and its views."""
        # print(f"Deleting PipelineManager UI (Tab Bar: {self._tab_bar_tag})")
        for view in self.task_views.values():
            view.delete()
        self.task_views.clear()
        if dpg.does_item_exist(self._tab_bar_tag):
            dpg.delete_item(self._tab_bar_tag)
