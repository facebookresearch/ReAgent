#!/usr/bin/env python3

import abc
from typing import Dict, NamedTuple, Optional

import ml.rl.types as mt
import torch
from caffe2.python import core, schema

from .normalization import MISSING_VALUE, NormalizationParameters
from .preprocessor_net import sort_features_by_normalization


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

    def create_const(self, init_net, name, value):
        blob = init_net.NextScopedBlob(name)
        init_net.GivenTensorFill([], blob, shape=[], values=[value])
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
    ) -> None:
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
        next_state = mt.FeatureVector(float_features=fetch(extract_record.next_state))
        action = fetch_action(extract_record.action)
        reward = fetch(input_record.reward)

        # is_terminal should be filled by preprocessor
        if self.max_q_learning:
            possible_next_actions = mt.PossibleActions(
                lengths=fetch(extract_record.possible_next_actions["lengths"]),
                actions=fetch_action(extract_record.possible_next_actions["values"]),
            )

            training_input = mt.MaxQLearningInput(
                state=state,
                action=action,
                next_state=next_state,
                possible_next_actions=possible_next_actions,
                reward=reward,
                is_terminal=None,
            )
        else:
            next_action = fetch_action(extract_record.next_action)
            training_input = mt.SARSAInput(
                state=state,
                action=action,
                next_state=next_state,
                next_action=next_action,
                reward=reward,
                is_terminal=None,
            )

        # TODO: stuff other fields in here
        extras = None

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
                ("next_state", next_state),
                (next_action_field, next_action_output),
            )
        )

        return FeatureExtractorNet(net, init_net)
