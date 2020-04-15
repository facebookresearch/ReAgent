#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Preprocess a sampled action (from the action_sampler) for input into the environment.
We assume the environment (gym.Env) expects a np.array as input.
Note that these actions are the ones inputted into RB, so we can assume here
that actor_output been called with cpu() and detach().
"""

import numpy as np
import reagent.types as rlt


def discrete_action_preprocessor(actor_output: rlt.ActorOutput) -> np.array:
    """ Simply reverses the one-hot encoding and convert to numpy """
    action = actor_output.action
    assert action.dim() == 1, "action has dim %d" % action.dim()
    idx = action.argmax().numpy()
    return idx


def continuous_action_preprocessor(actor_output: rlt.ActorOutput) -> np.array:
    """ Simply identity map """
    return actor_output.action.numpy()
