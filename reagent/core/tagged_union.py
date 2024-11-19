#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from reagent.core.fb_checker import IS_FB_ENVIRONMENT

if IS_FB_ENVIRONMENT:
    from fblearner.flow.core.types_lib.union import TaggedUnion as FlowTaggedUnion

    class TaggedUnion(FlowTaggedUnion):
        @classmethod
        def __get_validators__(cls):
            yield cls.pydantic_validate

        @classmethod
        def pydantic_validate(cls, v):
            if isinstance(v, cls):
                return v
            if not isinstance(v, dict):
                raise TypeError("Value should be dict")
            if len(v) != 1:
                raise ValueError(f"Expecting exactly one key, got {len(v)} keys.")
            key = list(v.keys())[0]
            if key not in cls.__annotations__:
                raise ValueError(f"Unknown key {key}")
            return cls(**{key: cls.__annotations__[key](**v[key])})

else:
    from dataclasses import fields

    class TaggedUnion:
        """
        Assuming that subclasses are pydantic's dataclass. All the fields must be Optional
        w/ None as default value. This doesn't support changing selected field/value.
        """

        @property
        def value(self):
            selected_fields = [
                field.name for field in fields(self) if getattr(self, field.name, None)
            ]
            assert (
                len(selected_fields) == 1
            ), f"{self} Expecting one selected field, got {selected_fields}"
            return getattr(self, selected_fields[0])
