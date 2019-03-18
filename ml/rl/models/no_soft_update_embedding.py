#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy

import torch.nn as nn


class NoSoftUpdateEmbedding(nn.Embedding):
    """
    Use this instead of vanilla Embedding module to avoid soft-updating the embedding
    table in the target network.
    """

    def __deepcopy__(self, memo):
        return copy.copy(self)
