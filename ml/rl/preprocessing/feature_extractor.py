#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from typing import Dict, List, NamedTuple, Optional

import ml.rl.types as mt
import numpy as np
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

    def read_actions_to_mask(
        self, net, name, num_actions, action, action_size_plus_one
    ):
        with core.NameScope(name):
            action_blob_one_hot = net.OneHot(
                [action(), action_size_plus_one], ["action_blob_one_hot"]
            )
            action_blob_one_hot_sliced = net.Slice(
                [action_blob_one_hot],
                ["action_blob_one_hot_sliced"],
                starts=[0, 0],
                ends=[-1, num_actions],
            )
        return action_blob_one_hot_sliced


def map_schema():
    return schema.Map(schema.Scalar(), schema.Scalar())


class InputColumn(object):
    STATE_FEATURES = "state_features"
    NEXT_STATE_FEATURES = "next_state_features"
    ACTION = "action"
    NEXT_ACTION = "next_action"
    POSSIBLE_ACTIONS = "possible_actions"
    POSSIBLE_ACTIONS_MASK = "possible_actions_mask"
    POSSIBLE_NEXT_ACTIONS = "possible_next_actions"
    POSSIBLE_NEXT_ACTIONS_MASK = "possible_next_actions_mask"
    NOT_TERMINAL = "not_terminal"
    STEP = "step"
    TIME_DIFF = "time_diff"
    MDP_ID = "mdp_id"
    SEQUENCE_NUMBER = "sequence_number"
    METRICS = "metrics"


class TrainingFeatureExtractor(FeatureExtractorBase):
    """
    Extract:
    - State
    - Action
    - Next state
    - Possible next actions/Next actions
    """

    def __init__(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None,
        include_possible_actions: bool = True,
        normalize: bool = True,
        max_num_actions: int = None,
        multi_steps: Optional[int] = None,
        metrics_to_score: Optional[List[str]] = None,
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
        self.include_possible_actions = include_possible_actions
        self.normalize = normalize
        self.max_num_actions = max_num_actions
        self.multi_steps = multi_steps
        self.metrics_to_score = metrics_to_score

    def extract(self, ws, input_record, extract_record):
        def fetch(b, to_torch=True):
            data = ws.fetch_blob(str(b()))
            if not isinstance(data, np.ndarray):
                # Blob uninitialized, return None and handle downstream
                return None
            if to_torch:
                return torch.tensor(data)
            return data

        def fetch_action(b):
            if self.sorted_action_features is None:
                return fetch(b)
            else:
                return mt.FeatureVector(float_features=fetch(b))

        def fetch_possible_actions(b):
            if self.sorted_action_features is not None:
                return mt.FeatureVector(float_features=fetch(b))
            else:
                return None

        state = mt.FeatureVector(float_features=fetch(extract_record.state_features))
        next_state = mt.FeatureVector(
            float_features=fetch(extract_record.next_state_features)
        )

        action = fetch_action(extract_record.action)
        next_action = fetch_action(extract_record.next_action)
        max_num_actions = None
        if self.multi_steps is not None:
            step = fetch(input_record.step).reshape(-1, 1)
        else:
            step = None
        reward = fetch(input_record.reward).reshape(-1, 1)

        # is_terminal should be filled by preprocessor
        not_terminal = fetch(input_record.not_terminal).reshape(-1, 1)
        time_diff = fetch(input_record.time_diff).reshape(-1, 1)

        if self.include_possible_actions:
            # TODO: this will need to be more complicated to support sparse features
            assert self.max_num_actions is not None, "Missing max_num_actions"
            possible_actions_mask = (
                fetch(extract_record.possible_actions_mask)
                .reshape(-1, self.max_num_actions)
                .type(torch.FloatTensor)
            )
            possible_next_actions_mask = fetch(
                extract_record.possible_next_actions_mask
            ).reshape(-1, self.max_num_actions)

            if self.sorted_action_features is not None:
                possible_actions = fetch_possible_actions(
                    extract_record.possible_actions
                )
                possible_next_actions = fetch_possible_actions(
                    extract_record.possible_next_actions
                )
                tiled_next_state = mt.FeatureVector(
                    float_features=next_state.float_features.repeat(
                        1, self.max_num_actions
                    ).reshape(-1, next_state.float_features.shape[1])
                )
                max_num_actions = self.max_num_actions

            else:
                possible_actions = None
                possible_next_actions = None
                tiled_next_state = None

            training_input = mt.MaxQLearningInput(
                state=state,
                action=action,
                next_state=next_state,
                tiled_next_state=tiled_next_state,
                possible_actions=possible_actions,
                possible_actions_mask=possible_actions_mask,
                possible_next_actions=possible_next_actions,
                possible_next_actions_mask=possible_next_actions_mask,
                next_action=next_action,
                reward=reward,
                not_terminal=not_terminal,
                step=step,
                time_diff=time_diff,
            )
        else:
            training_input = mt.SARSAInput(
                state=state,
                action=action,
                next_state=next_state,
                next_action=next_action,
                reward=reward,
                not_terminal=not_terminal,
                step=step,
                time_diff=time_diff,
            )

        mdp_id = fetch(input_record.mdp_id, to_torch=False)
        sequence_number = fetch(input_record.sequence_number)

        metrics = fetch(extract_record.metrics) if self.metrics_to_score else None

        # TODO: stuff other fields in here
        extras = mt.ExtraData(
            action_probability=fetch(input_record.action_probability).reshape(-1, 1),
            sequence_number=sequence_number.reshape(-1, 1)
            if sequence_number is not None
            else None,
            mdp_id=mdp_id.reshape(-1, 1) if mdp_id is not None else None,
            max_num_actions=max_num_actions,
            metrics=metrics,
        )

        return mt.TrainingBatch(training_input=training_input, extras=extras)

    def create_net(self):
        net = core.Net("feature_extractor")
        init_net = core.Net("feature_extractor_init")
        missing_scalar = self.create_const(init_net, "MISSING_SCALAR", MISSING_VALUE)

        action_schema = map_schema() if self.sorted_action_features else schema.Scalar()

        input_schema = schema.Struct(
            (InputColumn.STATE_FEATURES, map_schema()),
            (InputColumn.NEXT_STATE_FEATURES, map_schema()),
            (InputColumn.ACTION, action_schema),
            (InputColumn.NEXT_ACTION, action_schema),
            (InputColumn.NOT_TERMINAL, schema.Scalar()),
            (InputColumn.TIME_DIFF, schema.Scalar()),
        )
        if self.include_possible_actions:
            input_schema += schema.Struct(
                (InputColumn.POSSIBLE_ACTIONS_MASK, schema.List(schema.Scalar())),
                (InputColumn.POSSIBLE_NEXT_ACTIONS_MASK, schema.List(schema.Scalar())),
            )
            if self.sorted_action_features is not None:
                input_schema += schema.Struct(
                    (InputColumn.POSSIBLE_ACTIONS, schema.List(map_schema())),
                    (InputColumn.POSSIBLE_NEXT_ACTIONS, schema.List(map_schema())),
                )

        if self.metrics_to_score:
            input_schema += schema.Struct((InputColumn.METRICS, map_schema()))

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

        if self.sorted_action_features:
            action = self.extract_float_features(
                net,
                InputColumn.ACTION,
                input_record[InputColumn.ACTION],
                self.sorted_action_features,
                missing_scalar,
            )
            next_action = self.extract_float_features(
                net,
                InputColumn.NEXT_ACTION,
                input_record[InputColumn.NEXT_ACTION],
                self.sorted_action_features,
                missing_scalar,
            )
            if self.include_possible_actions:
                possible_action_features = self.extract_float_features(
                    net,
                    InputColumn.POSSIBLE_ACTIONS,
                    input_record[InputColumn.POSSIBLE_ACTIONS]["values"],
                    self.sorted_action_features,
                    missing_scalar,
                )
                possible_next_action_features = self.extract_float_features(
                    net,
                    InputColumn.POSSIBLE_NEXT_ACTIONS,
                    input_record[InputColumn.POSSIBLE_NEXT_ACTIONS]["values"],
                    self.sorted_action_features,
                    missing_scalar,
                )
        else:
            action_size_plus_one = self.create_const(
                init_net,
                "action_size_plus_one",
                self.max_num_actions + 1,
                dtype=core.DataType.INT64,
            )
            action = self.read_actions_to_mask(
                net,
                InputColumn.ACTION,
                self.max_num_actions,
                input_record[InputColumn.ACTION],
                action_size_plus_one,
            )
            next_action = self.read_actions_to_mask(
                net,
                InputColumn.NEXT_ACTION,
                self.max_num_actions,
                input_record[InputColumn.NEXT_ACTION],
                action_size_plus_one,
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
            if self.sorted_action_features is not None:
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
                if self.include_possible_actions:
                    possible_action_features, _ = PreprocessorNet().normalize_dense_matrix(
                        possible_action_features,
                        self.sorted_action_features,
                        self.action_normalization_parameters,
                        blobname_prefix="possible_action",
                        split_expensive_feature_groups=True,
                    )
                    possible_next_action_features, _ = PreprocessorNet().normalize_dense_matrix(
                        possible_next_action_features,
                        self.sorted_action_features,
                        self.action_normalization_parameters,
                        blobname_prefix="possible_next_action",
                        split_expensive_feature_groups=True,
                    )
            C2.set_net_and_init_net(None, None)

        if self.metrics_to_score:
            metrics_to_score_idxs = list(range(len(self.metrics_to_score)))
            missing_metric = self.create_const(init_net, "MISSING_METRIC", 0.0)
            metrics = self.extract_float_features(
                net,
                InputColumn.METRICS,
                input_record[InputColumn.METRICS],
                metrics_to_score_idxs,
                missing_metric,
            )

        output_schema = schema.Struct(
            (InputColumn.STATE_FEATURES, state),
            (InputColumn.NEXT_STATE_FEATURES, next_state),
            (InputColumn.ACTION, action),
            (InputColumn.NEXT_ACTION, next_action),
            (InputColumn.NOT_TERMINAL, input_record[InputColumn.NOT_TERMINAL]),
            (InputColumn.TIME_DIFF, input_record[InputColumn.TIME_DIFF]),
        )

        if self.include_possible_actions:
            # Drop the "lengths" blob from possible_actions_mask since we know
            # it's just a list of [max_num_actions, max_num_actions, ...]
            output_schema += schema.Struct(
                (
                    InputColumn.POSSIBLE_ACTIONS_MASK,
                    input_record[InputColumn.POSSIBLE_ACTIONS_MASK]["values"],
                ),
                (
                    InputColumn.POSSIBLE_NEXT_ACTIONS_MASK,
                    input_record[InputColumn.POSSIBLE_NEXT_ACTIONS_MASK]["values"],
                ),
            )
            if self.sorted_action_features is not None:
                output_schema += schema.Struct(
                    (InputColumn.POSSIBLE_ACTIONS, possible_action_features),
                    (InputColumn.POSSIBLE_NEXT_ACTIONS, possible_next_action_features),
                )

        if self.metrics_to_score:
            output_schema += schema.Struct((InputColumn.METRICS, metrics))

        net.set_output_record(output_schema)
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
