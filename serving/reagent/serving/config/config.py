#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from collections import OrderedDict
from typing import Dict, List, Union
from enum import Enum


class ConfigBaseMeta(type):
    def annotations_and_defaults(cls):
        annotations = OrderedDict()
        defaults = {}
        for base in reversed(cls.__bases__):
            if base is ConfigBase:
                continue
            annotations.update(getattr(base, "__annotations__", {}))
            defaults.update(getattr(base, "_field_defaults", {}))
        annotations.update(vars(cls).get("__annotations__", {}))
        defaults.update({k: getattr(cls, k) for k in annotations if hasattr(cls, k)})
        return annotations, defaults

    @property
    def __annotations__(cls):
        annotations, _ = cls.annotations_and_defaults()
        return annotations

    _field_types = __annotations__

    @property
    def _fields(cls):
        return cls.__annotations__.keys()

    @property
    def _field_defaults(cls):
        _, defaults = cls.annotations_and_defaults()
        return defaults


class ConfigBase(metaclass=ConfigBaseMeta):
    def items(self):
        return self._asdict().items()

    def _asdict(self):
        return {k: getattr(self, k) for k in type(self).__annotations__}

    def _replace(self, **kwargs):
        args = self._asdict()
        args.update(kwargs)
        return type(self)(**args)

    def __init__(self, **kwargs):
        """Configs can be constructed by specifying values by keyword.
      If a keyword is supplied that isn't in the config, or if a config requires
      a value that isn't specified and doesn't have a default, a TypeError will be
      raised."""
        specified = kwargs.keys() | type(self)._field_defaults.keys()
        required = type(self).__annotations__.keys()
        # Unspecified fields have no default and weren't provided by the caller
        unspecified_fields = required - specified
        if unspecified_fields:
            raise TypeError(f"Failed to specify {unspecified_fields} for {type(self)}")

        # Overspecified fields are fields that were provided but that the config
        # doesn't know what to do with, ie. was never specified anywhere.
        overspecified_fields = specified - required
        if overspecified_fields:
            raise TypeError(
                f"Specified non-existent fields {overspecified_fields} for {type(self)}"
            )

        vars(self).update(kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in sorted(self._asdict().items()):
            lines += f"{key}: {val}".split("\n")
        return "\n    ".join(lines)

    def __eq__(self, other):
        """Mainly a convenience utility for unit testing."""
        return type(self) == type(other) and self._asdict() == other._asdict()


class Operator(ConfigBase):
    name: str
    op_name: str
    input_dep_map: Dict[str, str]


ConstantValue = Union[
    str,
    int,
    float,
    List[str],
    List[int],
    List[float],
    Dict[str, str],
    Dict[str, int],
    Dict[str, float],
]


class Constant(ConfigBase):
    name: str
    value: ConstantValue


class DecisionRewardAggreation(Enum):
    DRA_INVALID = None,
    DRA_SUM = 'sum',
    DRA_MAX = 'max',


class DecisionConfig(ConfigBase):
    operators: List[Operator]
    constants: List[Constant]
    num_actions_to_choose: int
    reward_function: str
    reward_aggregator: DecisionRewardAggreation
