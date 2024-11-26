#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pkgutil


__all__ = []
print(list(pkgutil.walk_packages(__path__)))

for loader, module_name, _ in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    # pyre-fixme[16]: Item `MetaPathFinderProtocol` of `MetaPathFinderProtocol |
    #  PathEntryFinderProtocol` has no attribute `find_module`.
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module
