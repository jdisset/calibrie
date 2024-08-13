## {{{                          --     import     --
import shapely.geometry as sg
import dearpygui.dearpygui as dpg
from pathlib import Path
import xdialog
import calibry
import calibry.utils
import pandas as pd
from calibry.pipeline import Task
import typing
from typing import (
    Annotated,
    List,
    Callable,
    Any,
    get_type_hints,
    TypeVar,
    Type,
    Optional,
    ForwardRef,
    TypeAlias,
)
from pydantic import BaseModel, Field, computed_field, ConfigDict
from pydantic_core import PydanticUndefined
import xxhash
import base64
from numpy.typing import NDArray
import numpy as np
from functools import partial

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     base     --

unique_counter = 0


def unique_tag() -> str:
    global unique_counter
    unique_counter += 1
    random_salt = np.random.bytes(16)  # to avoid collisions with dpd'own tags
    data = f'{random_salt}.{unique_counter}'.encode('utf-8')
    hash_value = xxhash.xxh128(data).digest()
    return base64.b32encode(hash_value).decode('utf-8').rstrip('=')


class UIElement:
    def __init__(
        self,
        getter: Optional[Callable] = None,
        setter: Optional[Callable] = None,
        elt_type: Optional[Type] = None,
        in_component: Optional[Any] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.setter = setter
        self.getter = getter
        self.elt_type = elt_type
        self.in_component = in_component
        self._tag = unique_tag()

    def add(self, parent: int | str, **kwargs) -> int | str:
        if 'tag' in kwargs:
            self._tag = kwargs.pop('tag')
        raise NotImplementedError('UIElement.add must be implemented in derived classes')

    def delete(self):
        if dpg.does_item_exist(self._tag):
            dpg.delete_item(self._tag)


class UIEltSpec:
    def __init__(self, ui_type: Type[UIElement], **kwargs):
        self.ui_type = ui_type
        self.kwargs = kwargs


def get_first_UIEltSpec(metadata):
    for m in metadata:
        if isinstance(m, UIEltSpec):
            return m
    return None


class Component(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    def model_post_init(self, *args):
        super().model_post_init(*args)
        self._on_change_callbacks: dict[str, list[Callable[[Component, str], Any]]] = {}
        self._tag = unique_tag()
        if not hasattr(self, '_ui_fields_kwargs'):
            self._ui_fields_kwargs = {}
        if not hasattr(self, '_ui_grp_kwargs'):
            self._ui_grp_kwargs = {}

    def attr_changed(self, attr_name, old_value=None):
        pass

    def _add_ui_field(self, parent, uispec: UIEltSpec, field_name: str, **kwargs):
        def getter():
            return getattr(self, field_name)

        def setter(value):
            old_value = getattr(self, field_name)
            setattr(self, field_name, value)
            self.attr_changed(field_name, old_value)

        kw = {
            'tag': unique_tag(),
            'label': field_name,
            'default_value': getattr(self, field_name),
            'getter': getter,
            'setter': setter,
            **self._ui_fields_kwargs.get(field_name, {}),
            **uispec.kwargs,
            **kwargs,
        }

        uielt = uispec.ui_type(in_component=self, **kw)
        uielt.add(parent=parent)

    def add(self, parent, **kwargs) -> int | str:
        if 'tag' in kwargs:
            self._tag = kwargs.pop('tag')
        dpg.add_group(parent=parent, tag=self._tag, **self._ui_grp_kwargs)
        for field_name, field in self.model_fields.items():
            uispec = get_first_UIEltSpec(field.metadata)
            if uispec:
                self._add_ui_field(self._tag, uispec, field_name, **kwargs)
        return self._tag

    def delete(self):
        dpg.delete_item(self._tag)

    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        self._on_change(name, old_value)

    def register_on_change_callback(self, attr_name, callback, call_right_away=True):
        if attr_name not in self._on_change_callbacks:
            self._on_change_callbacks[attr_name] = []
        self._on_change_callbacks[attr_name].append(callback)
        if call_right_away:
            callback(self, attr_name)

    def _on_change(self, attr_name, old_value):
        if attr_name in self._on_change_callbacks:
            for callback in self._on_change_callbacks[attr_name]:
                callback(self, attr_name)





##────────────────────────────────────────────────────────────────────────────}}}
## {{{                    --     new component form     --


def new_component_form(
    parent,
    comp_type: Type[Component],
    form_submit_label='Create',
    form_callback: Optional[Callable[[Component], Any]] = None,
):
    make_dict = {}

    with dpg.group(parent=parent):
        for field_name, field in comp_type.model_fields.items():
            spec = get_first_UIEltSpec(field.metadata)
            if field.is_required() or spec:
                if not spec:
                    spec = UIEltSpec(AutoUI)

                def setter(value, field_name=field_name):
                    print(f'newcomopnent {field_name} setter = {value}')
                    make_dict[field_name] = value

                def getter(field_name=field_name):
                    return make_dict[field_name]

                kw = {
                    'tag': unique_tag(),
                    'label': field_name,
                    **spec.kwargs,
                }

                if 'readonly' in kw:
                    kw['readonly'] = False

                if field.default is not PydanticUndefined:
                    kw['default_value'] = field.default

                elt_type = field.annotation

                uielt = spec.ui_type(getter=getter, setter=setter, elt_type=elt_type, **kw)
                uielt.add(parent=parent)

        def validate(sender, app_data, user_data):
            try:
                constructed_obj = comp_type(**make_dict)
            except Exception as e:
                print(f'Error: {e}')
                with dpg.window(label="Error", modal=True, autosize=True, pos=[200, 200]):
                    dpg.add_text(str(e))
                return
            if form_callback is not None:
                form_callback(constructed_obj)

        dpg.add_button(label=form_submit_label, callback=validate)

    return make_dict


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     signatures     --


SIGNATURES = {
    'text': [
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
    ],
}


def filter_kwargs(sig, kwargs):
    return {k: v for k, v in kwargs.items() if k in sig}


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     Auto     --


def prepare_ui(ui, parent, **kwargs):
    assert ui.getter is not None, 'Getter not bound'
    assert ui.setter is not None, 'Setter not bound'

    kw = {
        **ui.kwargs,
        **kwargs,
    }

    if 'tag' in kw:
        ui._tag = kw.pop('tag')
        if not ui._tag:
            ui._tag = unique_tag()

    if 'in_component' in kw:
        ui.in_component = kw.pop('in_component')

    label = kw.get('label', '')

    def default_callback(sender, app_data, user_data):
        assert ui.setter is not None, 'Setter not bound'
        print(f'Callback: {app_data}')
        print(f'User data: {user_data}')
        print(f'ui: {ui}')
        print(f'ui.setter: {ui.setter}')
        ui.setter(dpg.get_value(ui._tag))

    kw = {
        'tag': ui._tag,
        'label': label,
        'callback': default_callback,
        **kw,
    }
    return kw


class DropdownUI(UIElement):
    def __init__(self, get_available_items: Callable, **kwargs):
        super().__init__(**kwargs)
        self.get_available_items = get_available_items

    def add(self, parent, **kwargs):
        kw = prepare_ui(self, parent, **kwargs)
        assert self.get_available_items is not None, 'get_available_items not bound'

        return dpg.add_combo(items=self.get_available_items(), **kw)


class TextUI(UIElement):
    def add(self, parent, **kwargs):
        kw = prepare_ui(self, parent, **kwargs)
        readonly = kw.pop('readonly', False)
        if readonly:
            return dpg.add_text(**filter_kwargs(SIGNATURES['text'], kw))
        else:
            return dpg.add_input_text(**kw)


class AutoUI(UIElement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._underlying = None

    def add(self, parent, **kwargs):
        kw = prepare_ui(self, parent, **kwargs)
        assert self.getter is not None, 'Getter not bound'

        if self.elt_type is None:
            # try to guess the type
            field_value = self.getter()
            self.elt_type = type(field_value)

        if issubclass(self.elt_type, str):
            self._underlying = TextUI(
                in_component=self.in_component,
                getter=self.getter,
                setter=self.setter,
                elt_type=str,
                **kw,
            )
            return self._underlying.add(parent)
        elif issubclass(self.elt_type, int):
            return dpg.add_input_int(**kw)
        elif issubclass(self.elt_type, float):
            return dpg.add_input_float(**kw)
        elif issubclass(self.elt_type, bool):
            return dpg.add_checkbox(**kw)
        elif issubclass(self.elt_type, Component):
            self._underlying = self.getter()
            return self._underlying.add(parent, **kwargs)
        else:
            raise ValueError(f'Field has unsupported type {self.elt_type}')

    def delete(self):
        if self._underlying is not None:
            self._underlying.delete()
        super().delete()


def make_ui_field(ui_type: Type[UIElement] = AutoUI, **kwargs):
    return UIEltSpec(ui_type, **kwargs)

def make_ui_elt(container, field_name, ui_type: Type[UIElement]=AutoUI, **kwargs):
    if not hasattr(container, '_ui_fields_kwargs'):
        container._ui_fields_kwargs = {}
    container._ui_fields_kwargs[field_name] = kwargs

    def getter():
        return getattr(container, field_name)

    def setter(value):
        setattr(container, field_name, value)

    return ui_type(in_component=container, getter=getter, setter=setter, **kwargs)

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                     --     new List Manager     --


class ListManager(UIElement):
    def __init__(
        self,
        item_type: Type[Component],
        item_ui_type: Type[UIElement] = AutoUI,
        add_button_label='+',
        add_button_callback: Optional[Callable] = None,
        on_delete_callback: Optional[Callable] = None,
        on_add_callback: Optional[Callable] = None,
        allow_duplicates: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.item_type = item_type
        self.item_ui_type = item_ui_type
        self.add_button_label = add_button_label
        self.on_delete_callback = on_delete_callback
        self.on_add_callback = on_add_callback
        self.allow_duplicates = allow_duplicates
        self._tag = unique_tag()
        self._ui_items: List = []
        self._item_tag_to_group = {}

        self.add_new_callback = self._create_new_item
        if add_button_callback is not None:

            def callback(sender, app_data, user_data):
                add_button_callback(self, sender, app_data, user_data)

            self.add_new_callback = callback

    def add(self, parent: int | str, **_):  # adds the manager to the parent container
        self.parent = parent
        assert self.getter is not None, 'Getter not bound'

        the_list = self.getter()
        list_copy = [e for e in the_list]
        the_list.clear()

        dpg.add_group(parent=self.parent, tag=self._tag)
        self._add_btn = dpg.add_button(
            label=self.add_button_label, callback=self.add_new_callback, parent=self._tag
        )

        with dpg.group(parent=self._tag):
            for item in list_copy:
                self.append(item)

        return self._tag

    def append(self, item):
        assert self.getter is not None, 'Getter not bound'
        this_list = self.getter()
        if not self.allow_duplicates:
            if item in this_list:
                return


        index = len(this_list)
        this_list.append(item)

        def it_getter():
            assert self.getter is not None, 'Getter not bound'
            this_list = self.getter()
            return this_list[index]

        def it_setter(value):
            assert self.getter is not None, 'Getter not bound'
            print(f'ListManager: setter {value}')
            this_list = self.getter()
            this_list[index] = value

        item_kwargs = {
            'getter': it_getter,
            'setter': it_setter,
        }

        item_ui = self.item_ui_type(**item_kwargs)

        with dpg.group(indent=1, parent=self._tag) as item_grp:
            del_button = dpg.add_button(
                label="-", callback=self._remove_item, user_data=index, parent=item_grp, tag=f'delbtn_{item_ui._tag}'
            )
            item_ui.add(parent=item_grp)
            self._item_tag_to_group[item_ui._tag] = item_grp

        self._ui_items.append(item_ui)

        if self.on_add_callback is not None:
            self.on_add_callback(self.in_component, item)

        return item_ui._tag

    def _create_new_item(self):
        new_item_window = dpg.add_window(
            label="New Item",
            modal=True,
            autosize=True,
            pos=[50, 50],
        )

        def new_item_callback(new_item):
            dpg.delete_item(new_item_window)
            self.append(new_item)

        new_component_form(
            new_item_window,
            self.item_type,
            form_submit_label='Add',
            form_callback=new_item_callback,
        )

    def _recompute_all_getters_and_setters(self):
        # assumes both lists are aligned and in-sync, but
        # the only that might be out of sync is the getter and setter
        # (i.e. order might have changed)
        assert self.getter is not None, 'Getter not bound'
        assert len(self._ui_items) == len(self.getter()), 'Length mismatch'
        for i, ui_item in enumerate(self._ui_items):

            def it_getter():
                assert self.getter is not None, 'Getter not bound'
                this_list = self.getter()
                return this_list[i]

            def it_setter(value):
                assert self.getter is not None, 'Getter not bound'
                this_list = self.getter()
                this_list[i] = value

            ui_item.getter = it_getter
            ui_item.setter = it_setter

        # also we need to update all the indexes in user_data of the delete buttons
        for i, ui_item in enumerate(self._ui_items):
            dpg.configure_item(f'delbtn_{ui_item._tag}', user_data=i)


    def _remove_item(self, sender, app_data, user_data):
        assert self.setter is not None, 'Setter not bound'
        index = user_data
        assert self.getter is not None, 'Getter not bound'
        this_list = self.getter()
        item = this_list[index]
        ui_item = self._ui_items[index]
        if self.on_delete_callback is not None:
            self.on_delete_callback(self.in_component, item, ui_item)
        del item

        grp_tag = self._item_tag_to_group.pop(ui_item._tag)
        if dpg.does_item_exist(grp_tag):
            dpg.delete_item(grp_tag)
        ui_item.delete()
        self._ui_items.pop(index)
        new_list = this_list[:index] + this_list[index + 1 :]
        self.setter(new_list)

        self._recompute_all_getters_and_setters()


##────────────────────────────────────────────────────────────────────────────}}}
