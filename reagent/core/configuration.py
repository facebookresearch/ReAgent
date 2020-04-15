#!/usr/bin/python3

import functools
from dataclasses import MISSING, Field, fields
from inspect import Parameter, signature
from typing import List, Optional, Type, Union

from reagent.core.dataclasses import dataclass
from torch import nn


BLACKLIST_TYPES = [nn.Module]


def make_config_class(
    func,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
    blacklist_types: List[Type] = BLACKLIST_TYPES,
):
    """
    Create a decorator to create dataclass with the arguments of `func` as fields.
    Only annotated arguments are converted to fields. If the default value is mutable,
    you must use `dataclass.field(default_factory=default_factory)` as default.
    In that case, the func has to be wrapped with @resolve_defaults below.

    `whitelist` & `blacklist` are mutually exclusive.
    """

    parameters = signature(func).parameters

    assert (
        whitelist is None or blacklist is None
    ), "whitelist & blacklist are mutually exclusive"

    blacklist_set = set(blacklist or [])

    def _is_type_blacklisted(t):
        if getattr(t, "__origin__", None) is Union:
            assert len(t.__args__) == 2 and t.__args__[1] == type(
                None
            ), "Only Unions of [X, None] (a.k.a. Optional[X]) are supported"
            t = t.__args__[0]
        return any(issubclass(t, blacklist_type) for blacklist_type in blacklist_types)

    whitelist = whitelist or [
        p.name
        for p in parameters.values()
        if p.name not in blacklist_set
        and p.annotation != Parameter.empty
        and not _is_type_blacklisted(p.annotation)
    ]

    for field_name in whitelist:
        p = parameters[field_name]
        assert p.annotation != Parameter.empty and not _is_type_blacklisted(
            p.annotation
        ), f"{field_name} has wrong annotation: {p.annotation}"

    def wrapper(config_cls):
        # Add __annotations__ for dataclass
        config_cls.__annotations__ = {
            field_name: parameters[field_name].annotation for field_name in whitelist
        }
        # Set default values
        for field_name in whitelist:
            default = parameters[field_name].default
            if default != Parameter.empty:
                setattr(config_cls, field_name, default)

        # Add hashing to support hashing list and dict
        config_cls.__hash__ = param_hash

        # Add non-recursive asdict(). dataclasses.asdict() is recursive
        def asdict(self):
            return {field.name: getattr(self, field.name) for field in fields(self)}

        config_cls.asdict = asdict

        return dataclass(frozen=True)(config_cls)

    return wrapper


def _resolve_default(val):
    if not isinstance(val, Field):
        return val
    if val.default != MISSING:
        return val.default
    if val.default_factory != MISSING:
        return val.default_factory()
    raise ValueError("No default value")


def resolve_defaults(func):
    """
    Use this decorator to resolve defualt field values in the constructor.
    """

    field_parameters = [
        p for p in signature(func).parameters.values() if isinstance(p.default, Field)
    ]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for p in field_parameters:
            if p.name not in kwargs:
                kwargs[p.name] = _resolve_default(p.default)
        return func(*args, **kwargs)

    return wrapper


def param_hash(p):
    """
    Use this to make parameters hashable. This is required because __hash__()
    is not inherited when subclass redefines __eq__(). We only need this when
    the parameter dataclass has a list or dict field.
    """
    return hash(tuple(_hash_field(getattr(p, f.name)) for f in fields(p)))


def _hash_field(val):
    """
    Returns hashable value of the argument. A list is converted to a tuple.
    A dict is converted to a tuple of sorted pairs of key and value.
    """
    if isinstance(val, list):
        return tuple(val)
    elif isinstance(val, dict):
        return tuple(sorted(val.items()))
    else:
        return val
