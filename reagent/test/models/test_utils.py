#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
import numpy.testing as npt


logger = logging.getLogger(__name__)


def check_save_load(
    self,
    model,
    expected_num_params,
    expected_num_inputs,
    expected_num_outputs,
    check_equality=True,
):
    # TODO: remove the expected_num* from call sites

    # TODO: revive this test or kill it
    # input_prototype = model.input_prototype()
    # traced_model = torch.jit.trace(model, input_prototype)

    # if check_equality:
    #     x = model(*input_prototype)
    #     y = traced_model(*input_prototype)
    #     self.assertEqual(x, y)

    pass
