import dearpygui.dearpygui as dpg
from calibrie.gui.core import (
    Component,
    UIElement,
    AutoUI,
    UIEltSpec,
    get_first_UIEltSpec,
    make_ui_field,
    unique_tag,
    PydanticUndefined,
    LoadedData,
    BaseModel,
    FilePickerUI,
    TextUI,
)
from calibrie.pipeline import Task
from typing import Annotated, Any, Optional, List, Dict, Union
from pathlib import Path
import inspect


from calibrie.gui.diagnostics import DiagnosticsPanel


# -- Add print function for easier debugging --
def _print_debug(msg, indent=0):
    print(f"{'  ' * indent}DEBUG: {msg}")


class TaskSettingsPanel(Component):
    """Displays and allows editing of task parameters."""

    task: Optional[Task] = None  # the task instance to configure

    def add(self, parent: str | int, task_instance: Task, **kwargs) -> str | int:
        """Adds the settings panel UI for the given task instance."""
        _print_debug(f"TaskSettingsPanel.add called for parent {parent}")
        self.task = task_instance
        if not self.task:
            _print_debug("  Task instance is None, cannot add settings panel.")
            dpg.add_text("No task selected", parent=parent)
            return ""

        task_name = getattr(self.task, '_name', f'task_{unique_tag()}')
        self._ui_tag = f"settings_{task_name}"
        _print_debug(f"  Task: {task_name} ({type(self.task).__name__})")
        _print_debug(f"  Panel Group Tag: {self._ui_tag}")

        if not dpg.does_item_exist(parent):
            _print_debug(f"  ERROR: Parent tag '{parent}' does not exist!")
            # Attempt to add to viewport as fallback? Or just fail?
            # For now, let it potentially fail later, but log the issue.

        # using try/except to ensure deletion happens even if add fails midway
        try:
            with dpg.group(parent=parent, tag=self._ui_tag, **kwargs):
                _print_debug(
                    f"  Created panel group '{self._ui_tag}' inside parent '{parent}'", indent=1
                )
                dpg.add_text(f"Settings: {task_name}", color=(200, 200, 200))
                dpg.add_separator()

                _print_debug(f"  Iterating through fields of {task_name}:", indent=1)
                if not hasattr(self.task, 'model_fields'):
                    _print_debug(
                        "  ERROR: Task instance has no 'model_fields'. Is it a Pydantic model?",
                        indent=2,
                    )
                    return self._ui_tag

                field_count = 0
                ui_added_count = 0
                for field_name, field in self.task.model_fields.items():
                    field_count += 1
                    _print_debug(f"  Field {field_count}: {field_name}", indent=2)

                    # basic filtering
                    if field_name.startswith('_') or field_name in [
                        'diagnostics_settings',
                        'priority',
                        'notes',
                    ]:
                        _print_debug(f"    Skipping (internal/control field).", indent=3)
                        continue

                    uispec = get_first_UIEltSpec(field.metadata)
                    field_type = field.annotation
                    _print_debug(f"    Annotation Type: {field_type}", indent=3)
                    _print_debug(f"    UI Spec found: {'Yes' if uispec else 'No'}", indent=3)

                    # determine the actual type to check
                    actual_type = field_type
                    origin = getattr(field_type, '__origin__', None)

                    if origin is Annotated:
                        actual_type = field_type.__args__[0]
                        origin = getattr(
                            actual_type, '__origin__', None
                        )  # check underlying type too

                    if origin is Union:  # handles Optional[T] and Union[A,B]
                        args = [
                            arg
                            for arg in getattr(actual_type, '__args__', [])
                            if arg is not type(None)
                        ]
                        if len(args) == 1:
                            actual_type = args[0]
                        else:
                            _print_debug(
                                f"    Complex Union type: {actual_type}, skipping auto UI.",
                                indent=3,
                            )
                            actual_type = None  # Can't auto-handle complex unions easily

                    _print_debug(f"    Actual Type for UI check: {actual_type}", indent=3)

                    should_add_ui = False
                    if uispec:
                        _print_debug(f"    Using explicit UI Spec: {uispec.ui_type}", indent=3)
                        should_add_ui = True
                    elif actual_type in [str, int, float, bool, Path, LoadedData]:
                        uispec = make_ui_field(AutoUI)  # default to AutoUI
                        _print_debug(f"    Using AutoUI for simple type.", indent=3)
                        should_add_ui = True
                    elif inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
                        default_val = getattr(self.task, field_name, field.default)
                        if isinstance(
                            default_val, Component
                        ):  # check if default is a Component we can add
                            uispec = make_ui_field(AutoUI)
                            _print_debug(f"    Using AutoUI for Component instance.", indent=3)
                            should_add_ui = True
                        else:
                            _print_debug(
                                f"    Skipping Pydantic model field without Component instance.",
                                indent=3,
                            )
                    else:
                        _print_debug(
                            f"    Skipping field: No UI spec and type not auto-handled.", indent=3
                        )

                    if should_add_ui:
                        try:
                            self._add_ui_field(self._ui_tag, uispec, field_name, self.task)
                            ui_added_count += 1
                        except Exception as e:
                            _print_debug(f"    ERROR adding UI for {field_name}: {e}", indent=3)
                            import traceback

                            traceback.print_exc()  # print full traceback for the error

                _print_debug(
                    f"  Finished field iteration. Added {ui_added_count} UI elements for {field_count} fields.",
                    indent=1,
                )

        except Exception as e:
            _print_debug(f"  ERROR creating panel group or iterating fields: {e}")
            import traceback

            traceback.print_exc()
            # ensure tag exists even if group creation failed, maybe add a text error?
            if not dpg.does_item_exist(self._ui_tag):
                dpg.add_text(
                    f"Error creating settings panel for {task_name}",
                    parent=parent,
                    tag=self._ui_tag,
                    color=(255, 0, 0),
                )

        _print_debug(f"TaskSettingsPanel.add finished for {task_name}")
        return self._ui_tag

    def _add_ui_field(
        self, parent_tag: str | int, uispec: UIEltSpec, field_name: str, task_instance: Task
    ):
        """Modified to work directly with the task_instance."""
        _print_debug(f"    _add_ui_field: Adding '{field_name}' to parent '{parent_tag}'", indent=4)
        field = task_instance.model_fields[field_name]

        ui_type = uispec.ui_type
        if ui_type == "AutoUI_Placeholder":
            ui_type = AutoUI  # defined in core
        _print_debug(f"      UI Type: {ui_type.__name__}", indent=5)

        # --- Define getter and setter ---
        def getter():
            val = getattr(task_instance, field_name)
            # _print_debug(f"        Getter for '{field_name}': returning {val} ({type(val)})", indent=6)
            return val

        def setter(value):
            _print_debug(
                f"        Setter for '{field_name}': received {value} ({type(value)})", indent=6
            )
            # ... (conversion logic as before) ...
            current_type = field.annotation
            expected_type = None

            if hasattr(current_type, '__origin__') and current_type.__origin__ is Annotated:
                expected_type = current_type.__args__[0]
            elif hasattr(current_type, '__origin__') and current_type.__origin__ is Union:
                args = [arg for arg in current_type.__args__ if arg is not type(None)]
                if len(args) == 1:
                    expected_type = args[0]
                else:
                    expected_type = current_type
            elif inspect.isclass(current_type):
                expected_type = current_type

            converted_value = value
            if expected_type and not isinstance(value, expected_type):
                try:
                    _print_debug(f"          Attempting conversion to {expected_type}", indent=7)
                    if expected_type is Path and isinstance(value, str):
                        converted_value = Path(value)
                    elif expected_type is bool and isinstance(value, (int, str)):
                        converted_value = (
                            bool(value) if isinstance(value, int) else value.lower() == 'true'
                        )
                    elif expected_type is LoadedData and isinstance(value, str):
                        converted_value = value
                    elif expected_type in [int, float, str]:
                        converted_value = expected_type(value)
                    _print_debug(
                        f"          Conversion result: {converted_value} ({type(converted_value)})",
                        indent=7,
                    )
                except Exception as conv_e:
                    _print_debug(f"          WARNING: Conversion failed: {conv_e}", indent=7)
                    converted_value = value
            try:
                setattr(task_instance, field_name, converted_value)
                _print_debug(f"          Successfully set attribute '{field_name}'", indent=7)
            except Exception as e:
                _print_debug(f"          ERROR setting attribute '{field_name}': {e}", indent=7)
                import traceback

                traceback.print_exc()

        # --- Prepare kwargs ---
        combined_kwargs = {
            'label': field_name,
            **uispec.kwargs,
        }
        default_val = getattr(task_instance, field_name, field.default)
        if default_val is not PydanticUndefined:
            combined_kwargs['default_value'] = default_val
        _print_debug(f"      Combined Kwargs: {combined_kwargs}", indent=5)

        # --- Instantiate and add ---
        ui_element = ui_type(
            getter=getter,
            setter=setter,
            elt_type=field.annotation,
            in_component=self,  # the settings panel is the component
            **combined_kwargs,
        )
        _print_debug(
            f"      Instantiated UI Element: {ui_element} with tag {ui_element._tag}", indent=5
        )

        # check parent exists just before adding
        if not dpg.does_item_exist(parent_tag):
            _print_debug(
                f"      ERROR: Parent '{parent_tag}' does not exist before adding element '{field_name}'!",
                indent=5,
            )
            return  # don't attempt to add

        try:
            added_tag = ui_element.add(parent=parent_tag)
            _print_debug(f"      Called ui_element.add(). Returned tag: {added_tag}", indent=5)
            if not dpg.does_item_exist(added_tag):
                _print_debug(
                    f"      WARNING: Element '{field_name}' (tag: {added_tag}) does not exist after adding!",
                    indent=5,
                )

        except Exception as add_e:
            _print_debug(
                f"      ERROR during ui_element.add() for '{field_name}': {add_e}", indent=5
            )
            import traceback

            traceback.print_exc()

    def delete(self):
        """Deletes the settings panel group."""
        _print_debug(f"TaskSettingsPanel.delete called for tag {self._ui_tag}")
        if dpg.does_item_exist(self._ui_tag):
            try:
                # recursively delete children first? dpg might handle this
                dpg.delete_item(self._ui_tag, children_only=False)
                _print_debug(f"  Deleted DPG item {self._ui_tag}")
            except Exception as e:
                _print_debug(f"  Error deleting DPG item {self._ui_tag}: {e}")
        else:
            _print_debug(f"  DPG item {self._ui_tag} did not exist.")
        self.task = None


class TaskView(Component):
    """Represents the view for a single task in a tab."""

    task: Optional[Task] = None
    settings_panel: Optional[TaskSettingsPanel] = None
    diagnostics_panel: Optional[DiagnosticsPanel] = None
    pipeline_manager: Optional[Any] = None  # link back to the manager

    _init_button_tag: str = ""
    _diag_button_tag: str = ""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.task = None
        self.settings_panel = TaskSettingsPanel()
        self.diagnostics_panel = DiagnosticsPanel()
        self._ui_tag = unique_tag()  # instance's unique group tag
        self._init_button_tag = unique_tag()
        self._diag_button_tag = unique_tag()
        self.pipeline_manager = None  # must be set after creation

    def add(
        self, parent: str | int, task_instance: Task, pipeline_manager: Any, **kwargs
    ) -> str | int:
        """Adds the task view UI."""
        self.task = task_instance
        self.pipeline_manager = pipeline_manager
        # use the tag generated in model_post_init
        view_tag = self._ui_tag

        # remove tag from kwargs if present before spreading
        local_kwargs = kwargs.copy()
        local_kwargs.pop('tag', None)

        # main group for the TaskView
        with dpg.group(parent=parent, tag=view_tag, **local_kwargs):  # use local_kwargs here
            # layout: left panel for settings, right for diagnostics + controls
            with dpg.group(horizontal=True):
                # left panel (settings)
                # use a child window for potential scrolling if settings are long
                with dpg.child_window(width=350, border=False):
                    self.settings_panel.add(dpg.last_item(), task_instance=self.task)

                # right panel (diagnostics & controls)
                with dpg.child_window(border=False):
                    # task control buttons
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label=f"Initialize '{self.task._name}'",
                            tag=self._init_button_tag,
                            callback=self._run_task_init,
                        )
                        dpg.add_button(
                            label=f"Run '{self.task._name}' Diagnostics",
                            tag=self._diag_button_tag,
                            callback=self._run_task_diagnostics,
                        )
                    dpg.add_separator()
                    # diagnostics area
                    self.diagnostics_panel.add(dpg.last_item())

        return view_tag

    def _run_task_init(self):
        """Callback to initialize the specific task."""
        if not self.task or not self.pipeline_manager:
            return
        print(f"Initializing task: {self.task._name}")
        try:
            # requires careful context handling - use full pipeline context for now
            context = self.pipeline_manager.get_context_for_task(self.task._name)
            new_context_data = self.task.initialize(context)
            self.pipeline_manager.update_context(new_context_data)
            print(f"Task {self.task._name} initialized successfully.")
            # maybe run diagnostics automatically after init?
            self._run_task_diagnostics()
        except Exception as e:
            print(f"Error initializing task {self.task._name}: {e}")
            # show error popup?
            import traceback

            traceback.print_exc()

    def _run_task_diagnostics(self):
        """Callback to run diagnostics for the specific task."""
        if not self.task or not self.pipeline_manager:
            return
        print(f"Running diagnostics for task: {self.task._name}")
        try:
            # use the potentially updated context from the manager
            context = self.pipeline_manager.context
            # ensure context is passed correctly if diagnostics need it
            figures = self.task.diagnostics(context, **self.task.diagnostics_settings)
            if figures:
                print(f"  Generated {len(figures)} figures.")
                # ensure source_task is set
                for fig in figures:
                    fig.source_task = self.task._name
                self.diagnostics_panel.set_figures(figures)
            else:
                print("  No diagnostic figures generated.")
                self.diagnostics_panel.set_figures([])

        except Exception as e:
            print(f"Error running diagnostics for task {self.task._name}: {e}")
            self.diagnostics_panel.set_figures([])  # clear panel on error
            import traceback

            traceback.print_exc()

    def delete(self):
        """Deletes the task view and its panels."""
        if self.settings_panel:
            self.settings_panel.delete()
        if self.diagnostics_panel:
            self.diagnostics_panel.delete()
        if dpg.does_item_exist(self._ui_tag):
            dpg.delete_item(self._ui_tag)
        self.task = None
        self.pipeline_manager = None
