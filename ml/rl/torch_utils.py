#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from io import BytesIO

import torch


def rescale_torch_tensor(
    tensor: torch.Tensor,
    new_min: torch.Tensor,
    new_max: torch.Tensor,
    prev_min: torch.Tensor,
    prev_max: torch.Tensor,
):
    """
    Rescale column values in N X M torch tensor to be in new range.
    Each column m in input tensor will be rescaled from range
    [prev_min[m], prev_max[m]] to [new_min[m], new_max[m]]
    """
    assert tensor.shape[1] == new_min.shape[1] == new_max.shape[1]
    assert tensor.shape[1] == prev_min.shape[1] == prev_max.shape[1]
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((tensor - prev_min) / prev_range) * new_range + new_min


def stack(mems):
    """
    Stack a list of tensors
    Could use torch.stack here but torch.stack is much slower
    than torch.cat + view
    Submitted an issue for investigation:
    https://github.com/pytorch/pytorch/issues/22462

    FIXME: Remove this function after the issue above is resolved
    """
    shape = (-1, *mems[0].shape)
    return torch.cat(mems).view(*shape)


def export_module_to_buffer(module) -> BytesIO:
    # traced_script_module = torch.jit.trace(module, module.input_prototype())
    write_buffer = BytesIO()
    torch.jit.save(module, write_buffer)
    return write_buffer
