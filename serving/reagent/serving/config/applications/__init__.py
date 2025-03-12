#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib
import pkgutil


__all__ = []
print(list(pkgutil.walk_packages(__path__)))

for _, module_name, _ in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = _module
