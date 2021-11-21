#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

from reagent.workflow.types import ModuleNameToEntityId


def get_workflow_id() -> int:
    # This is just stub. You will want to replace this file.
    return 987654321


def get_new_named_entity_ids(module_names: List[str]) -> ModuleNameToEntityId:
    result = {}
    i = 1
    done_one = False
    for name in module_names:
        if not done_one:
            result[name] = get_workflow_id()
            done_one = True
        else:
            # this is just random, you'll want to replace
            result[name] = 987654321 - i
            i += 1
    return result
