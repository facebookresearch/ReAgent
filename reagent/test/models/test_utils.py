#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
from typing import Callable, Optional

import torch
from reagent.models.base import ModelBase


logger = logging.getLogger(__name__)


def run_model_jit_trace(
    model: ModelBase,
    script_model,
    compare_func: Optional[Callable] = None,
):
    input_prototype = model.input_prototype()
    if not isinstance(input_prototype, (list, tuple)):
        input_prototype = (input_prototype,)
    tensor_input_prototype = tuple(x.float_features for x in input_prototype)
    traced_model = torch.jit.trace(script_model, tensor_input_prototype)

    x = model(*input_prototype)
    y = traced_model(*tensor_input_prototype)

    if compare_func:
        compare_func(x, y)
    elif isinstance(x, (list, tuple)):
        assert isinstance(y, (list, tuple))
        for xx, yy in x, y:
            assert isinstance(xx, torch.Tensor)
            assert isinstance(yy, torch.Tensor)
            assert torch.all(xx == yy)
    else:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert torch.all(x == y)
