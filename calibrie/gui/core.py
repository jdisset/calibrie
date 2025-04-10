import dearpygui.dearpygui as dpg
import numpy as np
import xxhash
import base64
from pydantic import BaseModel, Field, ConfigDict
from pydantic_core import PydanticUndefined
from functools import partial
import calibrie
import xdialog
from pathlib import Path
import pandas as pd
from calibrie.utils import ArbitraryModel, LoadedData  # Assuming LoadedData is defined here
import inspect
from pydantic import BaseModel, Field, ConfigDict
from pydantic_core import PydanticUndefined
from typing import (
    Optional,
    Callable,
    Any,
    Type,
    List,
    Dict,
    TypeVar,
    Annotated,
    Union,
)
import json  # for dict editor placeholder

unique_counter = 0
T = TypeVar('T')


def unique_tag() -> str:
    """Generates a unique tag for DPG items."""
    global unique_counter
    unique_counter += 1
    random_salt = np.random.bytes(4)
    data = f'{random_salt}-{unique_counter}'.encode('utf-8')
    hash_value = xxhash.xxh32(data).digest()
    return base64.urlsafe_b64encode(hash_value).decode('utf-8').rstrip('=')


class UIElement:
    """Base class for UI elements that bridge Pydantic fields and DPG items."""

    def __init__(
        self,
        getter: Optional[Callable] = None,
        setter: Optional[Callable] = None,
        elt_type: Optional[Type] = None,  # the python type of the field
        in_component: Optional[Any] = None,  # the Component instance this belongs to
        **kwargs,
    ):
        self.kwargs = kwargs
        self.setter = setter
        self.getter = getter
        self.elt_type = elt_type
        self.in_component = in_component
        self._tag = unique_tag()

    def add(self, parent: int | str, **kwargs) -> int | str:
        """Adds the DPG item to the parent."""
        if 'tag' in kwargs:
            self._tag = kwargs.pop('tag')
        if dpg.does_item_exist(self._tag):
            # print(f"Warning: Item with tag {self._tag} already exists. Reusing.")
            return self._tag
        kw = self.prepare_kwargs(**kwargs)
        dpg.add_group(parent=parent, tag=self._tag, **kw)
        return self._tag

    def delete(self):
        """Deletes the DPG item."""
        if dpg.does_item_exist(self._tag):
            try:
                # use children_only=False to delete the item itself too
                dpg.delete_item(self._tag, children_only=False)
            except Exception as e:
                print(f"Error deleting item {self._tag}: {e}")

    def prepare_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepares kwargs for DPG item creation, handling tag and callbacks."""
        if self.getter is None:
            raise AssertionError(f"Getter not bound for {self.__class__.__name__}")
        if self.setter is None:
            raise AssertionError(f"Setter not bound for {self.__class__.__name__}")

        kw = {
            **self.kwargs,
            **kwargs,
        }

        if 'tag' in kw:
            new_tag = kw.pop('tag')
            if new_tag:
                self._tag = new_tag
        if not self._tag:
            self._tag = unique_tag()

        if 'in_component' in kw:
            self.in_component = kw.pop('in_component')

        label = kw.get('label', '')

        def default_callback(sender, app_data, user_data):
            if self.setter is None:
                return  # should not happen due to assert above
            try:
                new_value = dpg.get_value(sender)
                self.setter(new_value)
                if hasattr(self.in_component, '_notify_change'):
                    field_name = kw.get('label')
                    if field_name:
                        self.in_component._notify_change(field_name)
            except Exception as e:
                print(f"Error in callback for {self._tag}: {e}")
                import traceback

                traceback.print_exc()

        final_kw = {
            'tag': self._tag,
            'label': label,
            'callback': default_callback,
            'user_data': self,
            **kw,
        }
        final_kw.pop('getter', None)
        final_kw.pop('setter', None)
        return final_kw


class UIEltSpec:
    """Specification for creating a UIElement."""

    def __init__(self, ui_type: Type[UIElement], **kwargs):
        self.ui_type = ui_type
        self.kwargs = kwargs


def get_first_UIEltSpec(metadata: tuple) -> Optional[UIEltSpec]:
    """Extracts the first UIEltSpec from field metadata."""
    for m in metadata:
        if isinstance(m, UIEltSpec):
            return m
    return None


# --- Base Component ---


class Component(ArbitraryModel):
    """Base model for components that can be rendered in the GUI."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    _ui_elements: Dict[str, UIElement] = {}  # maps field name to UIElement instance
    _ui_tag: str = ""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._ui_elements = {}
        self._ui_tag = unique_tag()  # group tag for this component's UI

    def add(self, parent: str | int, **kwargs) -> str | int:
        """Adds the component's UI representation to the parent DPG item."""
        if 'tag' in kwargs:
            self._ui_tag = kwargs.pop('tag')

        # main group for the component
        with dpg.group(parent=parent, tag=self._ui_tag, **kwargs):
            for field_name, field in self.model_fields.items():
                # skip internal fields
                if field_name.startswith('_'):
                    continue

                uispec = get_first_UIEltSpec(field.metadata)
                # use AutoUI placeholder mechanism for dynamic resolution
                if not uispec:
                    uispec = make_ui_field()  # defaults to AutoUI placeholder

                self._add_ui_field(self._ui_tag, uispec, field_name)

        return self._ui_tag

    def _add_ui_field(self, parent_tag: str | int, uispec: UIEltSpec, field_name: str, **kwargs):
        """Helper to create and add a UI element for a specific field."""
        field = self.model_fields[field_name]

        # resolve AutoUI placeholder if needed
        ui_type = uispec.ui_type
        if ui_type == "AutoUI_Placeholder":
            ui_type = AutoUI  # AutoUI should be defined by now

        def getter():
            return getattr(self, field_name)

        def setter(value):
            try:
                setattr(self, field_name, value)
                self._notify_change(field_name)
            except Exception as e:
                print(f"Error setting {field_name} to {value}: {e}")

        combined_kwargs = {
            'label': field_name,
            **uispec.kwargs,
            **kwargs,  # allow overriding via add method
        }
        default_val = getattr(self, field_name, field.default)
        if default_val is not PydanticUndefined:
            combined_kwargs['default_value'] = default_val

        ui_element = ui_type(
            getter=getter,
            setter=setter,
            elt_type=field.annotation,
            in_component=self,
            **combined_kwargs,
        )
        ui_element.add(parent=parent_tag)
        self._ui_elements[field_name] = ui_element

    def delete(self):
        """Deletes the component's UI group and all its child UI elements."""
        for field_name, ui_element in self._ui_elements.items():
            ui_element.delete()
        self._ui_elements.clear()

        if dpg.does_item_exist(self._ui_tag):
            try:
                dpg.delete_item(self._ui_tag)
            except Exception as e:
                print(f"Error deleting group {self._ui_tag}: {e}")

    def _notify_change(self, field_name: str):
        """Placeholder for reacting to changes originating from the GUI."""
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


# --- Specific UI Elements ---


class TextUI(UIElement):
    def add(self, parent: str | int, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        readonly = kw.pop('readonly', False)
        if 'default_value' in kw and kw['default_value'] is not None:
            kw['default_value'] = str(kw['default_value'])
        elif 'default_value' not in kw:
            kw['default_value'] = ""

        if readonly:
            text_sig = [
                'default_value',
                'label',
                'user_data',
                'use_internal_label',
                'tag',
                'indent',
                'parent',
                'before',
                'source',
                'payload_type',
                'drag_callback',
                'drop_callback',
                'show',
                'pos',
                'filter_key',
                'tracked',
                'track_offset',
                'wrap',
                'bullet',
                'color',
                'show_label',
            ]
            filtered_kw = {k: v for k, v in kw.items() if k in text_sig}
            # ensure parent is passed if not in filtered_kw (it's handled by prepare_kwargs)
            if 'parent' not in filtered_kw:
                filtered_kw['parent'] = parent
            return dpg.add_text(**filtered_kw)
        else:
            return dpg.add_input_text(parent=parent, **kw)


class IntUI(UIElement):
    def add(self, parent: str | int, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        if 'default_value' not in kw or kw['default_value'] is PydanticUndefined:
            kw['default_value'] = 0
        return dpg.add_input_int(parent=parent, **kw)


class FloatUI(UIElement):
    def add(self, parent: str | int, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        if 'default_value' not in kw or kw['default_value'] is PydanticUndefined:
            kw['default_value'] = 0.0
        return dpg.add_input_float(parent=parent, **kw)


class CheckboxUI(UIElement):
    def add(self, parent: str | int, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        if 'default_value' not in kw or kw['default_value'] is PydanticUndefined:
            kw['default_value'] = False
        return dpg.add_checkbox(parent=parent, **kw)


class DropdownUI(UIElement):
    def __init__(self, get_available_items: Callable[[], List[str]], **kwargs):
        super().__init__(**kwargs)
        self.get_available_items = get_available_items

    def add(self, parent: int | str, **kwargs) -> int | str:
        kw = self.prepare_kwargs(**kwargs)
        items = self.get_available_items()
        if 'default_value' not in kw or kw['default_value'] not in items:
            kw['default_value'] = items[0] if items else ""

        return dpg.add_combo(items=items, parent=parent, **kw)


class FilePickerUI(UIElement):
    def add(self, parent: str | int, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        is_dir = kw.pop('directory_selector', False)
        filetypes = kw.pop('filetypes', [])

        current_path = kw.get('default_value', "")
        if (
            isinstance(current_path, pd.DataFrame)
            and hasattr(current_path, 'attrs')
            and 'from_file' in current_path.attrs
        ):
            current_path = str(current_path.attrs['from_file'])
        elif not isinstance(current_path, str):
            current_path = ""

        with dpg.group(parent=parent, horizontal=True, tag=kw['tag']):
            # use group for layout, keep original label if exists
            dpg.add_text(self.kwargs.get('label', 'File/Dir:'))  # use original label from kwargs
            dpg.add_button(label="Browse...", tag=f"{kw['tag']}_button", callback=self._browse)
            dpg.add_input_text(
                default_value=current_path, readonly=True, tag=f"{kw['tag']}_display", width=-1
            )
        return kw['tag']

    def _browse(self, sender, app_data, user_data):
        is_dir = self.kwargs.get('directory_selector', False)
        filetypes = self.kwargs.get('filetypes', [])
        title = self.kwargs.get('label', 'Select File/Directory')

        if is_dir:
            path = xdialog.directory(title=title)
        else:
            path = xdialog.open_file(title=title, filetypes=filetypes)

        if path:
            dpg.set_value(f"{self._tag}_display", path)
            if self.setter is None:
                return
            try:
                self.setter(path)
                if hasattr(self.in_component, '_notify_change'):
                    field_name = self.kwargs.get('label')
                    if field_name:
                        self.in_component._notify_change(field_name)
            except Exception as e:
                print(f"Error setting path via setter for {self._tag}: {e}")


# --- List Manager UI ---
class ListManagerUI(UIElement):
    """Basic UI for managing a list of simple strings."""

    def add(self, parent: int | str, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        label = kw.get('label', 'List')

        # main group for the list manager
        with dpg.group(parent=parent, tag=self._tag):
            dpg.add_text(label)
            # add button to add new items
            dpg.add_button(label="+", callback=self._add_item_callback, user_data=self._tag)
            # child window to hold the list items, makes it scrollable
            with dpg.child_window(height=100, border=True, tag=f"{self._tag}_list_area"):
                self._rebuild_list_items()

        return self._tag

    def _rebuild_list_items(self):
        """Clears and rebuilds the list items in the UI."""
        list_area_tag = f"{self._tag}_list_area"
        if not dpg.does_item_exist(list_area_tag):
            return

        # delete existing children in the list area
        dpg.delete_item(list_area_tag, children_only=True)

        current_list = self.getter() if self.getter else []
        if not isinstance(current_list, list):
            # handle non-list case (e.g., None or wrong type)
            dpg.add_text("Invalid list data", parent=list_area_tag, color=(255, 0, 0))
            return

        for index, item in enumerate(current_list):
            with dpg.group(horizontal=True, parent=list_area_tag):
                # for now, assume items are strings
                dpg.add_input_text(
                    default_value=str(item),
                    width=-50,
                    callback=self._edit_item_callback,
                    user_data=(index, self._tag),
                )
                dpg.add_button(
                    label="-",
                    width=-1,
                    callback=self._remove_item_callback,
                    user_data=(index, self._tag),
                )

    def _add_item_callback(self, sender, app_data, user_data):
        """Adds a new default item to the list."""
        if not self.getter or not self.setter:
            return
        current_list = self.getter()[:]  # make a copy
        # add a default value (e.g., empty string)
        current_list.append("")
        self.setter(current_list)
        self._rebuild_list_items()  # update ui

    def _edit_item_callback(self, sender, app_data, user_data):
        """Edits an existing item in the list."""
        if not self.getter or not self.setter:
            return
        index, manager_tag = user_data
        current_list = self.getter()[:]  # make a copy
        if 0 <= index < len(current_list):
            # app_data is the new string value from add_input_text
            current_list[index] = app_data
            self.setter(current_list)
            # no need to rebuild here unless order changes

    def _remove_item_callback(self, sender, app_data, user_data):
        """Removes an item from the list."""
        if not self.getter or not self.setter:
            return
        index, manager_tag = user_data
        current_list = self.getter()[:]  # make a copy
        if 0 <= index < len(current_list):
            current_list.pop(index)
            self.setter(current_list)
            self._rebuild_list_items()  # update ui


# --- Dict Editor UI (Placeholder) ---
class DictEditorUI(UIElement):
    """Basic placeholder UI for dictionary fields."""

    def add(self, parent: int | str, **kwargs) -> str | int:
        kw = self.prepare_kwargs(**kwargs)
        label = kw.get('label', 'Dictionary')
        current_dict = self.getter() if self.getter else {}

        with dpg.group(parent=parent, tag=self._tag):
            dpg.add_text(label)
            # display dict as json string in a read-only text input
            try:
                dict_str = json.dumps(current_dict, indent=2)
            except TypeError:
                dict_str = f"Cannot display dictionary of type: {type(current_dict)}"

            # use input text with multiline for better display
            dpg.add_input_text(
                default_value=dict_str, multiline=True, readonly=True, width=-1, height=100
            )
            dpg.add_text("(Dictionary editing not implemented yet)", color=(200, 200, 0))

        return self._tag


# --- AutoUI ---
class AutoUI(UIElement):
    # updated map to include list and dict
    _type_map: Dict[Type, Type[UIElement]] = {
        str: TextUI,
        int: IntUI,
        float: FloatUI,
        bool: CheckboxUI,
        Path: FilePickerUI,
        LoadedData: FilePickerUI,
        list: ListManagerUI,  # map list to ListManagerUI
        dict: DictEditorUI,  # map dict to DictEditorUI
        Component: lambda **kwargs: kwargs['default_value'],
    }

    def add(self, parent: str | int, **kwargs) -> str | int:
        specific_ui_type = None
        # handle annotated types properly
        actual_type = self.elt_type
        origin = getattr(self.elt_type, '__origin__', None)

        if origin is Annotated:
            actual_type = self.elt_type.__args__[0]
            origin = getattr(actual_type, '__origin__', None)  # check underlying type too
        if origin is Union:
            args = [arg for arg in getattr(actual_type, '__args__', []) if arg is not type(None)]
            if len(args) == 1:
                actual_type = args[0]
            else:
                actual_type = None  # cannot auto handle complex unions

        if actual_type:
            for py_type, ui_element_type in self._type_map.items():
                try:
                    if inspect.isclass(actual_type) and issubclass(actual_type, py_type):
                        specific_ui_type = ui_element_type
                        break
                    # handle list and dict origins directly
                    elif origin and issubclass(origin, py_type):
                        specific_ui_type = ui_element_type
                        break

                except TypeError:
                    pass  # issubclass fails if actual_type is not a class (like a Union)

        if specific_ui_type is None:
            # check if it's a component instance (value based)
            default_val = self.getter() if self.getter else None
            if isinstance(default_val, Component):
                specific_ui_type = AutoUI  # special case handled below
            else:
                print(
                    f"Warning: AutoUI couldn't determine widget for type {self.elt_type}. Defaulting to TextUI."
                )
                specific_ui_type = TextUI  # fallback

        # special case for rendering component instances directly
        if specific_ui_type is AutoUI and isinstance(self.getter(), Component):
            component_instance = self.getter()
            return component_instance.add(parent, **kwargs)  # directly add the component

        specific_element = specific_ui_type(
            getter=self.getter,
            setter=self.setter,
            elt_type=self.elt_type,
            in_component=self.in_component,
            **self.kwargs,
        )
        return specific_element.add(parent, **kwargs)


# -- Factory function --
def make_ui_field(ui_type: Type[UIElement] = AutoUI, **kwargs) -> UIEltSpec:
    """Factory function to create UIEltSpec annotations easily."""
    # handle placeholder case if AutoUI is passed before definition (though it should be defined now)
    if ui_type == "AutoUI_Placeholder":
        ui_type = AutoUI
    return UIEltSpec(ui_type, **kwargs)


# --- Task Registry ---
class TaskRegistry:
    _tasks: Dict[str, Type[calibrie.pipeline.Task]] = {}

    @classmethod
    def register(cls, task_class: Type[calibrie.pipeline.Task]):
        name = task_class.__name__
        if name in cls._tasks:
            print(f"Warning: Task '{name}' already registered. Overwriting.")
        cls._tasks[name] = task_class

    @classmethod
    def get_task_class(cls, name: str) -> Optional[Type[calibrie.pipeline.Task]]:
        return cls._tasks.get(name)

    @classmethod
    def get_registered_tasks(cls) -> Dict[str, Type[calibrie.pipeline.Task]]:
        return cls._tasks.copy()

    @classmethod
    def discover_tasks(cls):
        # print("Discovering tasks...") # removed verbose print
        for name, obj in inspect.getmembers(calibrie):
            if (
                inspect.isclass(obj)
                and issubclass(obj, calibrie.pipeline.Task)
                and obj is not calibrie.pipeline.Task
            ):
                cls.register(obj)


# TaskRegistry.discover_tasks()
