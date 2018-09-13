#!/usr/bin/env python3

"""
EXPERIMENTAL

These generic types define common interface between modules. Above all, these
facilitates model exporting through ONNX. ONNX doesn't trace dictionary so we
use NamedTuple to act in place of dictionaries. NamedTuple is also more compact
than dictionary; so, this should be good overall.

"""

from typing import NamedTuple, Optional, Union

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

Action = Union[DiscreteAction, ParametricAction]

State = FeatureVector


class StateAction(NamedTuple):
    state: State
    action: Action


class PossibleActions(NamedTuple):
    lengths: ValueType
    actions: Action


class StatePossibleActions(NamedTuple):
    state: State
    possible_actions: PossibleActions


class MaxQLearningInput(NamedTuple):
    state_action: StateAction
    state_possible_actions: StatePossibleActions
    reward: ValueType
    is_terminal: ValueType


class SARSAInput(NamedTuple):
    state_action: StateAction
    next_state_action: StateAction
    reward: ValueType
    is_terminal: ValueType
