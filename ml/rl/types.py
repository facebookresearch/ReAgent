#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
EXPERIMENTAL

These generic types define common interface between modules. Above all, these
facilitates model exporting through ONNX. ONNX doesn't trace dictionary so we
use NamedTuple to act in place of dictionaries. NamedTuple is also more compact
than dictionary; so, this should be good overall.

"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Type, TypeVar, Union

import numpy as np
import torch
from caffe2.python import core


"""
We use a mix of frameworks in our system. Therefore, we can't pinpoint the exact
type of value
"""
ValueType = Union[torch.Tensor, core.BlobReference, np.ndarray]


@dataclass
class IdFeatureConfig:
    """
    This describes how to map raw features to model features
    """

    feature_id: int  # integer feature ID
    id_mapping_name: str  # key to ModelPreprocessingConfig.id_mapping_config


@dataclass
class IdFeatureBase:
    """
    User should subclass this class and define each ID feature as a field w/ ValueType
    as the type of the field.
    """

    @classmethod
    # TODO: This should be marked as abstractmethod but mypi doesn't like it.
    # See https://github.com/python/mypy/issues/5374
    # @abc.abstractmethod
    def get_feature_config(cls) -> Dict[str, IdFeatureConfig]:
        """
        Returns mapping from feature name, which must be a field in this dataclass, to
        feature config.
        """
        raise NotImplementedError


T = TypeVar("T", bound="SequenceFeatureBase")


@dataclass
class FloatFeatureInfo:
    name: str
    feature_id: int


@dataclass
class SequenceFeatureBase:
    id_features: Optional[IdFeatureBase]
    float_features: Optional[ValueType]

    @classmethod
    # TODO: This should be marked as abstractmethod but mypi doesn't like it.
    # See https://github.com/python/mypy/issues/5374
    # @abc.abstractmethod
    def get_max_length(cls) -> int:
        """
        Subclass should return the max-length of this sequence. If the raw data is
        longer, feature extractor will truncate the front. If the raw data is shorter,
        feature extractor will fill the front with zero.
        """
        raise NotImplementedError

    @classmethod
    def get_float_feature_infos(cls) -> List[FloatFeatureInfo]:
        """
        Override this if the sequence has float features associated to it.
        Float features should be stored as ID-score-list, where the ID part corresponds
        to primary entity ID of the sequence. E.g., if this is a sequence of previously
        watched videos, then the key should be video ID.
        """
        return []

    @classmethod
    def prototype(cls: Type[T]) -> T:
        float_feature_infos = cls.get_float_feature_infos()
        float_features = (
            torch.rand(1, cls.get_max_length(), len(float_feature_infos))
            if float_feature_infos
            else None
        )
        fields = dataclasses.fields(cls)
        id_features = None
        for field in fields:
            if field.name != "id_features" or not isinstance(field.type, type):
                continue
            id_feature_fields = dataclasses.fields(field.type)
            id_features = field.type(  # noqa
                **{
                    f.name: torch.randint(1, (1, cls.get_max_length()))
                    for f in id_feature_fields
                }
            )
            break

        return cls(id_features=id_features, float_features=float_features)


U = TypeVar("U", bound="SequenceFeatures")


@dataclass
class SequenceFeatures:
    """
    A stub-class for sequence features in the model. All fileds should be subclass of
    SequenceFeatureBase above.
    """

    @classmethod
    def prototype(cls: Type[U]) -> U:
        fields = dataclasses.fields(cls)
        return cls(**{f.name: f.type.prototype() for f in fields})  # noqa


@dataclass
class IdMapping:
    ids: List[int]


@dataclass
class ModelFeatureConfig:
    float_feature_infos: List[FloatFeatureInfo]
    id_mapping_config: Dict[str, IdMapping]
    sequence_features_type: Optional[Type[SequenceFeatures]]


class FeatureVector(NamedTuple):
    float_features: Optional[ValueType] = None
    # sequence_features should ideally be Mapping[str, IdListFeature]; however,
    # that doesn't work well with ONNX.
    # User is expected to dynamically define the type of id_list_features based
    # on the actual features used in the model.
    sequence_features: Optional[SequenceFeatureBase] = None
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[ValueType] = None


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


class MemoryNetworkInput(NamedTuple):
    state: State
    action: Action
    next_state: ValueType
    reward: ValueType
    not_terminal: ValueType


class ExtraData(NamedTuple):
    mdp_id: Optional[ValueType] = None
    sequence_number: Optional[ValueType] = None
    action_probability: Optional[ValueType] = None
    max_num_actions: Optional[int] = None
    metrics: Optional[ValueType] = None


class TrainingBatch(NamedTuple):
    training_input: Union[MaxQLearningInput, SARSAInput, MemoryNetworkInput]
    extras: Any

    def __len__(self):
        return self.training_input.state.float_features.size()[0]


class SingleQValue(NamedTuple):
    q_value: ValueType


class AllActionQValues(NamedTuple):
    q_values: ValueType


class CappedContinuousAction(NamedTuple):
    """
    Continuous action in range [-1, 1], e.g., the output of DDPG actor
    """

    action: ValueType


class MemoryNetworkOutput(NamedTuple):
    mus: ValueType
    sigmas: ValueType
    logpi: ValueType
    reward: ValueType
    not_terminal: ValueType
    next_lstm_hidden: ValueType
    next_lstm_cell: ValueType
