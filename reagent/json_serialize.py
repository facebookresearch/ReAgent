#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import json
import logging
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, NamedTuple, Type, Union


logger = logging.getLogger(__name__)


def object_to_json(o: Any) -> str:
    assert is_dataclass(o), "Only dataclasses can be serialized"
    return json.dumps(prepare_for_json(o))


def prepare_for_json(o: Any) -> Any:
    if isinstance(o, NamedTuple):
        d = {}
        for field_name in o._fields:
            d[field_name] = prepare_for_json(getattr(o, field_name))
        return d
    elif is_dataclass(o):
        return asdict(o)
    else:
        return o


def json_to_object(j: str, to_type: Type) -> Any:
    assert is_dataclass(to_type), "Only dataclasses can be deserialized"
    j_obj = json.loads(j)
    return from_json(j_obj, to_type)


def from_json(j_obj: Any, to_type: Type) -> Any:
    if j_obj is None:
        return None
    logger.debug("TYPE: ")
    logger.debug(j_obj)
    logger.debug(to_type)
    if getattr(to_type, "_field_types", None) is not None:
        # Type is a NamedTuple, dive in
        field_data = {}
        for field_name in j_obj.keys():
            assert (
                field_name in to_type._fields
            ), "Item in dict missing from {}: {}".format(str(to_type), field_name)
            field_value = j_obj[field_name]
            object_type = to_type._field_types[field_name]
            if getattr(object_type, "__origin__", None) is Union:
                assert len(object_type.__args__) == 2 and object_type.__args__[
                    1
                ] == type(
                    None
                ), "Only Unions of [X, None] (a.k.a. Optional[X]) are supported"
                object_type = object_type.__args__[0]
            field_data[field_name] = from_json(field_value, object_type)
        return to_type(**field_data)  # Create the NamedTuple
    elif is_dataclass(to_type):
        # Type is a dataclass, dive in
        field_types = {}
        for field in fields(to_type):
            field_types[field.name] = field.type
        field_data = {}
        for field_name in j_obj.keys():
            assert field_name in field_types, "Item in dict missing from {}: {}".format(
                str(to_type), field_name
            )
            field_value = j_obj[field_name]
            object_type = field_types[field_name]
            if getattr(object_type, "__origin__", None) is Union:
                assert len(object_type.__args__) == 2 and object_type.__args__[
                    1
                ] == type(
                    None
                ), "Only Unions of [X, None] (a.k.a. Optional[X]) are supported"
                object_type = object_type.__args__[0]
            field_data[field_name] = from_json(field_value, object_type)
        return to_type(**field_data)  # Create the NamedTuple
    elif getattr(to_type, "_name", None) is not None and to_type._name == "List":
        assert isinstance(
            j_obj, list
        ), "Tried to set the wrong type to a list: {}".format(j_obj)
        list_inner_type = to_type.__args__[0]
        retval_list = []
        for i in j_obj:
            retval_list.append(from_json(i, list_inner_type))
        return retval_list
    elif getattr(to_type, "_name", None) is not None and to_type._name == "Dict":
        assert isinstance(
            j_obj, dict
        ), "Tried to set the wrong type to a dict: {}".format(j_obj)
        dict_inner_key_type = to_type.__args__[0]
        dict_inner_value_type = to_type.__args__[1]
        retval_dict = {}
        for k, v in j_obj.items():
            retval_dict[from_json(k, dict_inner_key_type)] = from_json(
                v, dict_inner_value_type
            )
        return retval_dict
    else:
        return j_obj
