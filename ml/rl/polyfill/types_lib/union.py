#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
from typing import Any, Dict, Optional, Tuple

from ml.rl.polyfill.decorators import classproperty
from ml.rl.polyfill.exceptions import NonRetryableTypeError


class _EmptyValue:
    def __bool__(self):
        return False

    def __nonzero__(self):
        return False


class _UnionObject:
    def __init__(self, **fields):
        # type: (**Any) -> None
        if len(fields) != 1:
            raise NonRetryableTypeError(
                "The constructor of {} must be called with"
                " exactly one key and value. (Got {})".format(
                    type(self).__name__, fields
                )
            )

        [(field, value)] = fields.items()
        self._set_field(field, value)

    @classproperty
    @classmethod
    def field_spec(cls):
        # type: () -> Dict
        raise NotImplementedError()

    @property
    def selected_field(self):
        # type: () -> str
        field, _ = self._get_field_value_pair()
        return field

    @property
    def value(self):
        # type: () -> Any
        _, value = self._get_field_value_pair()
        return value

    def _get_field(self, field):
        # type: (str) -> Any
        return self.__dict__["#"][field]

    def _get_field_value_pair(self):
        # type: () -> Tuple[str, Any]
        [(field, value)] = (
            (field, value)
            for field, value in self.__dict__["#"].items()
            if not isinstance(value, _EmptyValue)
        )
        return field, value

    def _clear_all_fields(self):
        # type: () -> None
        # pyre-fixme[18]: Global name `_EMPTY_VALUE` is undefined.
        # pyre-fixme[16]: Callable `field_spec` has no attribute `__iter__`.
        self.__dict__["#"] = {field: _EMPTY_VALUE for field in self.field_spec}

    def _set_field(self, field, value):
        # type: (str, Any) -> None
        # pyre-fixme[16]: Callable `field_spec` has no attribute `__getitem__`.
        self._clear_all_fields()
        self.__dict__["#"][field] = value

    def __getattr__(self, field):
        # type: (str) -> Any
        if field.startswith("__"):
            raise AttributeError("Unable to get the private field '{}'".format(field))
        if field == "field_spec":
            raise AttributeError(
                "'field_spec' is a classproperty, it probably failed with an AttributeError'"
            )
        if field not in self.field_spec:
            raise AttributeError("'{}' is not a valid field of {}".format(field, self))
        else:
            value = self._get_field(field)
            # pyre-fixme[18]: Global name `_EMPTY_VALUE` is undefined.
            if value is _EMPTY_VALUE:
                raise AttributeError("Field '{}' of {} is not set".format(field, self))
            else:
                return value

    def __setattr__(self, field, value):
        # type: (str, Any) -> None
        if field.startswith("__"):
            raise AttributeError("Unable to set the private field '{}'".format(field))
        if field not in self.field_spec:
            raise AttributeError("'{}' is not a valid field of {}".format(field, self))
        else:
            self._set_field(field, value)

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, _UnionObject):
            my_field, my_value = self._get_field_value_pair()
            their_field, their_value = other._get_field_value_pair()
            return (
                my_field == their_field
                and my_value == their_value
                and self.field_spec[my_field] == other.field_spec[their_field]
            )
        return False

    def __hash__(self):
        # type: () -> int
        field, value = self._get_field_value_pair()
        # TODO: call _hash_field(value)
        return hash((field, value))

    def __contains__(self, name):
        # type: (str) -> bool
        field, _ = self._get_field_value_pair()
        return name == field

    def __repr__(self):
        # type: () -> str
        field, value = self._get_field_value_pair()
        return "{}({}:{}|{})".format(
            self.__class__.__name__,
            field,
            value,
            # pyre-fixme[16]: Callable `field_spec` has no attribute `__iter__`.
            ",".join(sorted(str(field) for field in self.field_spec)),
        )


class TaggedUnion(_UnionObject):
    def __deepcopy__(self, memo):
        # type: (Optional[Dict]) -> TaggedUnion
        field, value = self._get_field_value_pair()
        fields = {field: value}
        return self.__class__(**copy.deepcopy(fields, memo))

    def walk_object(self, func):
        field, value = self._get_field_value_pair()
        return self.__class__(**{field: func(value)})

    @classproperty
    @classmethod
    def field_spec(cls):
        # type: () -> Dict
        return cls.__annotations__


_EMPTY_VALUE = _EmptyValue()
