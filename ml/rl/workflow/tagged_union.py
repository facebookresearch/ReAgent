#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
from typing import Any, Tuple


class _EmptyValue:
    def __bool__(self):
        return False

    def __nonzero__(self):
        return False

_EMPTY_VALUE = _EmptyValue()

class TaggedUnion:
    def __init__(self, **fields):
        if len(fields) != 1:
            raise TypeError(
                f"The constructor of {type(self).__name__} must be called with"
                " exactly one key and value. (Got {fields})"
            )

        [(field, value)] = fields.items()
        self._set_field(field, value)

    def __deepcopy__(self, memo):
        field, value = self._get_field_value_pair()
        fields = {field: value}
        return self.__class__(**copy.deepcopy(fields, memo))

    @property
    def selected_field(self) -> str:
        field, _ = self._get_field_value_pair()
        return field

    @property
    def value(self) -> Any:
        _, value = self._get_field_value_pair()
        return value

    def _get_field(self, field: str) -> Any:
        return self.__dict__["#"][field]

    def _get_field_value_pair(self) -> Tuple[str, Any]:
        [(field, value)] = (
            (field, value)
            for field, value in self.__dict__["#"].items()
            if not isinstance(value, _EmptyValue)
        )
        return field, value

    def _clear_all_fields(self) -> None:
        self.__dict__["#"] = {field: _EMPTY_VALUE for field in self.__annotations__}

    def _set_field(self, field: str, value: Any) -> None:
        if field not in self.__annotations__:
            # We can't just use self in the message because this could be called from ctor
            raise AttributeError(f"'{field}' is not a valid field of {type(self).__name__}")
        self._clear_all_fields()
        self.__dict__["#"][field] = value

    def __getattr__(self, field: str) -> Any:
        if field.startswith("__"):
            raise AttributeError(f"Unable to get the private field '{field}'")
        if field not in self.__annotations__:
            raise AttributeError("'{}' is not a valid field of {}".format(field, self))
        else:
            value = self._get_field(field)
            if value is _EMPTY_VALUE:
                raise AttributeError("Field '{}' of {} is not set".format(field, self))
            else:
                return value

    def __setattr__(self, field: str, value: Any) -> None:
        if field.startswith("__"):
            raise AttributeError("Unable to set the private field '{}'".format(field))
        if field not in self.__annotations__:
            raise AttributeError("'{}' is not a valid field of {}".format(field, self))
        else:
            self._set_field(field, value)

    def __eq__(self, other) -> bool:
        if isinstance(other, TaggedUnion):
            my_field, my_value = self._get_field_value_pair()
            their_field, their_value = other._get_field_value_pair()
            return (
                my_field == their_field
                and my_value == their_value
                and self.__annotations__[my_field] == other.__annotations__[their_field]
            )
        return False

    def __hash__(self) -> int:
        field, value = self._get_field_value_pair()
        return hash((field, value))

    def __contains__(self, name) -> bool:
        field, _ = self._get_field_value_pair()
        return name == field

    def __repr__(self) -> str:
        field, value = self._get_field_value_pair()
        return "{}({}:{}|{})".format(
            self.__class__.__name__,
            field,
            value,
            ",".join(sorted(str(field) for field in self.__annotations__)),
        )
