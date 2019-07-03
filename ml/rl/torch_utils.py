#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch


def stack(mems):
    """
    Stack a list of tensors
    Could use torch.stack here but torch.stack is much slower
    than torch.cat + view in buck dev mode
    """
    shape = (-1, *mems[0].shape)
    return torch.cat(mems).view(*shape)
