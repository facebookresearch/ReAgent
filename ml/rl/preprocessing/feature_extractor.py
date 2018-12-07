#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from typing import Dict, NamedTuple, Optional

import ml.rl.types as mt
import torch
from caffe2.python import core, schema
from ml.rl.caffe_utils import C2

from .normalization import (
    MISSING_VALUE,
    NormalizationParameters,
    sort_features_by_normalization,
)
from .preprocessor_net import PreprocessorNet


"""
These are set of classes facilitating interface between Caffe2-style data source
and PyTorch model
"""


class FeatureExtractorNet(NamedTuple):
    """
    `init_net` will only be run once. The external outputs of `init_net` are
    assumed to be parameters of `net` and will be saved in the predictor file.
    `net` should not use any parameters not initialized by `init_net`.
    """

    net: core.Net
    init_net: core.Net


class FeatureExtractorBase(object, metaclass=abc.ABCMeta):
    """
    This is not a transformer because Caffe2 has a weird API. We cannot directly
    call functions. It's easier to handle separately.
    """

    def __init__(self, ws=None):
        super(FeatureExtractorBase, self).__init__()

    def extract(self, ws, input_record, extract_record):
        """
        If the extractor is to be run, e.g., by the reader, then subclass should
        implement

        Args:
            input_record (schema.Field): the record given to the net
            extract_record (schema.Field): the output the net
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_net(self) -> FeatureExtractorNet:
        """
        Returns FeatureExtractorNet to perform feature extraction.

        The returned net must have input & output record set so that it can be
        bound to actual inputs/outputs
        """
        pass

    def create_const(self, init_net, name, value, dtype=core.DataType.FLOAT):
        blob = init_net.NextScopedBlob(name)
        if not isinstance(value, list):
            shape = []
            value = [value]
        else:
            shape = [len(value)]
        init_net.GivenTensorFill([], blob, shape=shape, values=value, dtype=dtype)
        init_net.AddExternalOutput(blob)
        return blob

    def extract_float_features(self, net, name, field, keys_to_extract, missing_scalar):
        """
        Helper function to extract matrix from stacked  sparse tensors
        """
        with core.NameScope(name):
            float_features, _presence_mask = net.SparseToDenseMask(
                [field.keys(), field.values(), missing_scalar, field.lengths()],
                ["float_features", "presence_mask"],
                mask=keys_to_extract,
            )
        return float_features


def map_schema():
    return schema.Map(schema.Scalar(), schema.Scalar())


class InputColumn(object):
    STATE_FEATURES = "state_features"
    NEXT_STATE_FEATURES = "next_state_features"
    ACTION = "action"
    NEXT_ACTION = "next_action"
    POSSIBLE_NEXT_ACTIONS = "possible_next_actions"


class TrainingFeatureExtractor(FeatureExtractorBase):
    """
    Extract:
    - State
    - Action
    - Next state
    - Possible next actions/Next actions (depending on max_q_learning)
    """

    def __init__(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None,
        max_q_learning: bool = True,
        normalize: bool = True,
    ) -> None:
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.sorted_state_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        if action_normalization_parameters:
            self.sorted_action_features, _ = sort_features_by_normalization(
                action_normalization_parameters
            )
        else:
            self.sorted_action_features = None
        self.max_q_learning = max_q_learning
        self.normalize = normalize

    def extract(self, ws, input_record, extract_record):
        def fetch(b):
            data = ws.fetch_blob(str(b()))
            return torch.tensor(data)

        def fetch_action(b):
            if self.sorted_action_features is None:
                return fetch(b)
            else:
                return mt.FeatureVector(float_features=fetch(b))

        state = mt.FeatureVector(float_features=fetch(extract_record.state))
        action = fetch_action(extract_record.action)
        reward = fetch(input_record.reward).reshape(-1, 1)

        # is_terminal should be filled by preprocessor
        if self.max_q_learning:
            if self.sorted_action_features is not None:
                next_state = None
                tiled_next_state = mt.FeatureVector(
                    float_features=fetch(extract_record.tiled_next_state)
                )
            else:
                next_state = mt.FeatureVector(
                    float_features=fetch(extract_record.next_state)
                )
                tiled_next_state = None
            possible_next_actions = mt.PossibleActions(
                lengths=fetch(extract_record.possible_next_actions["lengths"]),
                actions=fetch_action(extract_record.possible_next_actions["values"]),
            )

            training_input = mt.MaxQLearningInput(
                state=state,
                action=action,
                next_state=next_state,
                tiled_next_state=tiled_next_state,
                possible_next_actions=possible_next_actions,
                reward=reward,
                not_terminal=(possible_next_actions.lengths > 0).float().reshape(-1, 1),
            )
        else:
            next_state = mt.FeatureVector(
                float_features=fetch(extract_record.next_state)
            )
            next_action = fetch_action(extract_record.next_action)
            training_input = mt.SARSAInput(
                state=state,
                action=action,
                next_state=next_state,
                next_action=next_action,
                reward=reward,
                # HACK: Need a better way to check this
                not_terminal=torch.ones_like(reward),
            )

        # TODO: stuff other fields in here
        extras = mt.ExtraData(
            action_probability=fetch(input_record.action_probability).reshape(-1, 1)
        )

        return mt.TrainingBatch(training_input=training_input, extras=extras)

    def create_net(self):
        net = core.Net("feature_extractor")
        init_net = core.Net("feature_extractor_init")
        missing_scalar = self.create_const(init_net, "MISSING_SCALAR", MISSING_VALUE)

        action_schema = map_schema() if self.sorted_action_features else schema.Scalar()

        if self.max_q_learning:
            next_action_field = InputColumn.POSSIBLE_NEXT_ACTIONS
            next_action_schema = schema.List(action_schema)
        else:
            next_action_field = InputColumn.NEXT_ACTION
            next_action_schema = action_schema

        input_schema = schema.Struct(
            (InputColumn.STATE_FEATURES, map_schema()),
            (InputColumn.NEXT_STATE_FEATURES, map_schema()),
            (InputColumn.ACTION, action_schema),
            (next_action_field, next_action_schema),
        )

        input_record = net.set_input_record(input_schema)

        state = self.extract_float_features(
            net,
            "state",
            input_record[InputColumn.STATE_FEATURES],
            self.sorted_state_features,
            missing_scalar,
        )
        next_state = self.extract_float_features(
            net,
            "next_state",
            input_record[InputColumn.NEXT_STATE_FEATURES],
            self.sorted_state_features,
            missing_scalar,
        )

        if self.max_q_learning and self.sorted_action_features is not None:
            next_state_field = "tiled_next_state"
            # TODO: this will need to be more complicated to support sparse features
            next_state = net.LengthsTile(
                [next_state, input_record.possible_next_actions.lengths()],
                ["tiled_next_state"],
            )
        else:
            next_state_field = "next_state"

        action = input_record.action
        next_action = input_record[next_action_field]
        if self.max_q_learning:
            next_action = next_action["values"]
        if self.sorted_action_features:
            action = self.extract_float_features(
                net, "action", action, self.sorted_action_features, missing_scalar
            )
            next_action = self.extract_float_features(
                net,
                next_action_field,
                next_action,
                self.sorted_action_features,
                missing_scalar,
            )

        if self.normalize:
            C2.set_net_and_init_net(net, init_net)
            state, _ = PreprocessorNet().normalize_dense_matrix(
                state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="state",
                split_expensive_feature_groups=True,
            )
            next_state, _ = PreprocessorNet().normalize_dense_matrix(
                next_state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="next_state",
                split_expensive_feature_groups=True,
            )
            if self.sorted_action_features:
                action, _ = PreprocessorNet().normalize_dense_matrix(
                    action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="action",
                    split_expensive_feature_groups=True,
                )
                next_action, _ = PreprocessorNet().normalize_dense_matrix(
                    next_action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="next_action",
                    split_expensive_feature_groups=True,
                )
            C2.set_net_and_init_net(None, None)

        next_action_output = (
            schema.List(
                next_action, lengths_blob=input_record.possible_next_actions.lengths
            )
            if self.max_q_learning
            else next_action
        )

        net.set_output_record(
            schema.Struct(
                ("state", state),
                ("action", action),
                (next_state_field, next_state),
                (next_action_field, next_action_output),
            )
        )

        return FeatureExtractorNet(net, init_net)


class PredictorFeatureExtractor(FeatureExtractorBase):
    """
    This class assumes that action is not in the input unless it's parametric
    action.

    The features (of both states & actions, if any) are expected to come in the
    following blobs:
    - input/float_features.keys
    - input/float_features.values
    - input/float_features.lengths

    TODO: Support int features
    """

    def __init__(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None,
        normalize: bool = True,
    ) -> None:
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.sorted_state_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        if action_normalization_parameters:
            self.sorted_action_features, _ = sort_features_by_normalization(
                action_normalization_parameters
            )
        else:
            self.sorted_action_features = None
        self.normalize = normalize

    def extract(self, ws, input_record, extract_record):
        def fetch(b):
            data = ws.fetch_blob(str(b()))
            return torch.tensor(data)

        state = mt.FeatureVector(float_features=fetch(extract_record.state))
        if self.sorted_action_features is None:
            action = None
        else:
            action = mt.FeatureVector(float_features=fetch(extract_record.action))
        return mt.StateAction(state=state, action=action)

    def create_net(self):
        net = core.Net("feature_extractor")
        init_net = core.Net("feature_extractor_init")
        missing_scalar = self.create_const(init_net, "MISSING_SCALAR", MISSING_VALUE)

        input_schema = schema.Struct(
            (
                "float_features",
                schema.Map(
                    keys=core.BlobReference("input/float_features.keys"),
                    values=core.BlobReference("input/float_features.values"),
                    lengths_blob=core.BlobReference("input/float_features.lengths"),
                ),
            )
        )

        input_record = net.set_input_record(input_schema)

        state = self.extract_float_features(
            net,
            "state",
            input_record.float_features,
            self.sorted_state_features,
            missing_scalar,
        )

        if self.sorted_action_features:
            action = self.extract_float_features(
                net,
                "action",
                input_record.float_features,
                self.sorted_action_features,
                missing_scalar,
            )

        if self.normalize:
            C2.set_net_and_init_net(net, init_net)
            state, _ = PreprocessorNet().normalize_dense_matrix(
                state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="state",
                split_expensive_feature_groups=True,
            )
            if self.sorted_action_features:
                action, _ = PreprocessorNet().normalize_dense_matrix(
                    action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="action",
                    split_expensive_feature_groups=True,
                )
            C2.set_net_and_init_net(None, None)

        output_record = schema.Struct(("state", state))
        if self.sorted_action_features:
            output_record += schema.Struct(("action", action))

        net.set_output_record(output_record)

        return FeatureExtractorNet(net, init_net)
