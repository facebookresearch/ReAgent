#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


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
        assert len(selected_fields) == 1, f"Expecting one selected field"
        return getattr(self, selected_fields[0])
