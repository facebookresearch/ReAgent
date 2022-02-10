#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.fb_checker import IS_FB_ENVIRONMENT


if IS_FB_ENVIRONMENT:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor  # noqa
else:
    # TODO: KeyedJaggedTensor/JaggedTensor are dummy classes in OSS
    # We haven't been able to install torchrec properly in OSS as of Jan 2022
    class KeyedJaggedTensor:
        def __init__(self, keys=None, lengths=None, values=None, weights=None):
            self._weights = None

        def __getitem__(self, x):
            pass

        def keys(self):
            pass

        def values(self):
            pass

        @classmethod
        def concat(cls, a, b):
            pass

    class JaggedTensor:
        def __init__(self):
            self._weights = None

        def values(self):
            pass

        def lengths(self):
            pass
