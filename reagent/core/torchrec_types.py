#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from enum import Enum

from reagent.core.fb_checker import IS_FB_ENVIRONMENT

if IS_FB_ENVIRONMENT:
    from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
    from torchrec import PoolingType
    from torchrec.models.dlrm import SparseArch, InteractionArch  # noqa
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
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

    class PoolingType(Enum):
        MEAN = "mean"
        SUM = "sum"

    class EmbeddingBagConfig:
        def __init__(
            self, name, feature_names, num_embeddings, embedding_dim, pooling=None
        ):
            self.embedding_dim = embedding_dim

    class EmbeddingBagCollection:
        def __init__(self, device, tables):
            self.embedding_bag_configs = []
            pass

    class SparseArch:
        def __init__(self, embedding_bag_collection):
            pass

    class InteractionArch:
        def __init__(self, num_sparse_features):
            pass
