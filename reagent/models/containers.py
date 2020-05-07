#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn
from reagent.models.base import ModelBase


class Sequential(
    nn.Sequential,  # type: ignore
    ModelBase,
):
    """
    Used this instead of torch.nn.Sequential to automate model tracing
    """

    def input_prototype(self):
        first = self[0]
        assert isinstance(
            first, ModelBase
        ), "The first module of Sequential has to be ModelBase"
        return first.input_prototype()
