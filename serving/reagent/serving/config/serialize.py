#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
from enum import Enum
from typing import Dict, List, Tuple, Union


def _get_class_type(cls):
    """
    type(cls) has an inconsistent behavior between 3.6 and 3.7 because of
    changes in the typing module. We therefore rely on __extra (3.6) and
    __origin__ (3.7), present only in classes from typing to extract the origin
    of the class for comparison, otherwise default to the type sent directly
    :param cls: class to infer
    :return: class or in the case of classes from typing module, the real type
    (Union, List) of the created object
    """
    return getattr(cls, "__extra__", getattr(cls, "__origin__", cls))


def _is_optional(cls):
    return _get_class_type(cls) == Union and type(None) in cls.__args__


def _extend_tuple_type(cls, value):
    sub_cls_list = list(cls.__args__)
    if len(sub_cls_list) != len(value):
        if len(sub_cls_list) != 2 or sub_cls_list[1] is not Ellipsis:
            raise ValueError(
                f"{len(value)} values found which is more than number of types in tuple {cls}"
            )
        del sub_cls_list[1]
        sub_cls_list.extend((cls.__args__[0],) * (len(value) - len(sub_cls_list)))
    return sub_cls_list


def _value_to_json(cls, value):
    cls_type = _get_class_type(cls)
    assert _is_optional(cls) or value is not None
    if value is None:
        return value
    elif _is_optional(cls) and len(cls.__args__) == 2:
        sub_cls = cls.__args__[0] if type(None) != cls.__args__[0] else cls.__args__[1]
        return _value_to_json(sub_cls, value)
    elif cls_type == Union:
        real_cls = type(value)
        if hasattr(real_cls, "_fields"):
            value = config_to_json(real_cls, value)

        if issubclass(real_cls, str):
            name = "string_value"
        elif issubclass(real_cls, int):
            name = "int_value"
        elif issubclass(real_cls, float):
            name = "double_value"
        elif issubclass(real_cls, List):
            if len(value) == 0:
                raise TypeError(
                    f"Can not properly decide the type of {value} since it is empty"
                )
            real_cls = type(value[0])
            if issubclass(real_cls, str):
                name = "list_string_value"
            elif issubclass(real_cls, int):
                name = "list_int_value"
            elif issubclass(real_cls, float):
                name = "list_double_value"
            else:
                raise TypeError(f"{real_cls} type not supported")
        elif issubclass(real_cls, Dict):
            if len(value) == 0:
                raise TypeError(
                    f"Can not properly decide the type of {value} since it is empty"
                )
            items = list(value.items())
            key_real_cls, value_real_cls = type(items[0][0]), type(items[0][1])
            if not issubclass(key_real_cls, str):
                raise TypeError(f"{value} type not supported")
            if issubclass(value_real_cls, str):
                name = "map_string_value"
            elif issubclass(value_real_cls, int):
                name = "map_int_value"
            elif issubclass(value_real_cls, float):
                name = "map_double_value"
            else:
                raise TypeError(f"{real_cls} type not supported")
        else:
            raise TypeError(f"{real_cls} type not supported")
        return {name: value}
    elif hasattr(cls, "_fields"):
        return config_to_json(cls, value)
    elif issubclass(cls_type, Enum):
        return value.value
    elif issubclass(cls_type, List):
        sub_cls = cls.__args__[0]
        return [_value_to_json(sub_cls, v) for v in value]
    elif issubclass(cls_type, Tuple):
        return tuple(
            _value_to_json(c, v) for c, v in zip(_extend_tuple_type(cls, value), value)
        )
    elif issubclass(cls_type, Dict):
        sub_cls = cls.__args__[1]
        return {key: _value_to_json(sub_cls, v) for key, v in value.items()}
    return value


def config_to_json(cls, config_obj):
    json_result = {}
    if not hasattr(cls, "_fields"):
        raise Exception(f"{cls} is not a valid config class")
    for field, f_cls in cls.__annotations__.items():
        value = getattr(config_obj, field)
        json_result[field] = _value_to_json(f_cls, value)
    return json_result
