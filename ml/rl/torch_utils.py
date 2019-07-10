#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from io import BytesIO

import torch


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
