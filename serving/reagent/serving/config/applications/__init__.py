#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-strict

import importlib
import pkgutil
import types


__all__: list[str] = []
print(list(pkgutil.walk_packages(__path__)))

for _, module_name, _ in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module: types.ModuleType = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = _module
