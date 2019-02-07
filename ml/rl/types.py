#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
EXPERIMENTAL

These generic types define common interface between modules. Above all, these
facilitates model exporting through ONNX. ONNX doesn't trace dictionary so we
use NamedTuple to act in place of dictionaries. NamedTuple is also more compact
than dictionary; so, this should be good overall.

"""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
import torch
from caffe2.python import core


"""
We use a mix of frameworks in our system. Therefore, we can't pinpoint the exact
type of value
"""
ValueType = Union[torch.Tensor, core.BlobReference, np.ndarray]


class IdListFeature(NamedTuple):
    lengths: ValueType
    values: ValueType


class FeatureVector(NamedTuple):
    float_features: Optional[ValueType] = None
    # id_list_features should ideally be Mapping[str, IdListFeature]; however,
    # that doesn't work well with ONNX.
    # User is expected to dynamically define the type of id_list_features based
    # on the actual features used in the model.
    id_list_features: Optional[NamedTuple] = None


DiscreteAction = ValueType

ParametricAction = FeatureVector


class ActorOutput(NamedTuple):
    action: ValueType
    log_prob: Optional[ValueType] = None


Action = Union[
    DiscreteAction, ParametricAction
]  # One-hot vector for discrete action DQN and feature vector for everyone else

State = FeatureVector


class StateInput(NamedTuple):
    """
    This class makes it easier to plug modules into predictor
    """

    state: State


class StateAction(NamedTuple):
    state: State
    action: Action


class MaxQLearningInput(NamedTuple):
    state: State
    action: Action
    next_action: Action
    next_state: Optional[State]  # Available in case of discrete action
    tiled_next_state: Optional[State]  # Available in case of parametric action
    possible_actions: Optional[Action]
    possible_actions_mask: ValueType
    possible_next_actions: Optional[Action]
    possible_next_actions_mask: ValueType
    reward: ValueType
    not_terminal: ValueType
    step: Optional[ValueType]
    time_diff: ValueType


class SARSAInput(NamedTuple):
    state: State
    action: Action
    next_state: State
    next_action: Action
    reward: ValueType
    not_terminal: ValueType
    step: Optional[ValueType]
    time_diff: ValueType


class ExtraData(NamedTuple):
    mdp_id: Optional[ValueType] = None
    sequence_number: Optional[ValueType] = None
    action_probability: Optional[ValueType] = None
    max_num_actions: Optional[int] = None


class TrainingBatch(NamedTuple):
    training_input: Union[MaxQLearningInput, SARSAInput]
    extras: Any


class SingleQValue(NamedTuple):
    q_value: ValueType


class AllActionQValues(NamedTuple):
    q_values: ValueType


class CappedContinuousAction(NamedTuple):
    """
    Continuous action in range [-1, 1], e.g., the output of DDPG actor
    """

    action: ValueType
