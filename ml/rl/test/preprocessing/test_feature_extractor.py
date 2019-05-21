#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.testing as npt
import torch
from caffe2.python import schema, workspace
from ml.rl import types as rlt
from ml.rl.preprocessing.feature_extractor import (
    PredictorFeatureExtractor,
    TrainingFeatureExtractor,
    WorldModelFeatureExtractor,
    id_list_schema,
    id_score_list_schema,
    map_schema,
)
from ml.rl.preprocessing.identify_types import CONTINUOUS, PROBABILITY
from ml.rl.preprocessing.normalization import MISSING_VALUE, NormalizationParameters
from ml.rl.test.base.utils import (
    ABIdFeatures,
    CIdFeatures,
    FloatOnlySequence,
    IdAndFloatSequence,
    IdOnlySequence,
    NumpyFeatureProcessor,
    SequenceFeatures,
)


class FeatureExtractorTestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_state_normalization_parameters(self):
        return {
            i: NormalizationParameters(
                feature_type=PROBABILITY if i % 2 else CONTINUOUS, mean=0, stddev=1
            )
            for i in range(1, 5)
        }

    def get_action_normalization_parameters(self):
        # Sorted order: 12, 11, 13
        return {
            i: NormalizationParameters(
                feature_type=CONTINUOUS if i % 2 else PROBABILITY, mean=0, stddev=1
            )
            for i in range(11, 14)
        }

    def get_time_since_first_normalization_parameters(self):
        return NormalizationParameters(feature_type=CONTINUOUS, mean=0, stddev=1)

    def setup_state_features(self, ws, field):
        lengths = np.array([3, 0, 5], dtype=np.int32)
        keys = np.array([2, 1, 9, 1, 2, 3, 4, 5], dtype=np.int64)
        values = np.arange(8).astype(np.float32)
        ws.feed_blob(str(field.lengths()), lengths)
        ws.feed_blob(str(field.keys()), keys)
        ws.feed_blob(str(field.values()), values)
        return lengths, keys, values

    def expected_state_features(self, normalize):
        # Feature order: 1, 3, 2, 4
        dense = np.array(
            [
                [1, MISSING_VALUE, 0, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [3, 5, 4, 6],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [1, 3, 2, 4], self.get_state_normalization_parameters()
            )
        return dense

    def setup_next_state_features(self, ws, field):
        lengths = np.array([2, 2, 4], dtype=np.int32)
        keys = np.array([2, 1, 9, 1, 2, 3, 4, 5], dtype=np.int64)
        values = np.arange(10, 18).astype(np.float32)
        ws.feed_blob(str(field.lengths()), lengths)
        ws.feed_blob(str(field.keys()), keys)
        ws.feed_blob(str(field.values()), values)
        return lengths, keys, values

    def expected_next_state_features(self, normalize):
        # Feature order: 1, 3, 2, 4
        dense = np.array(
            [
                [11, MISSING_VALUE, 10, MISSING_VALUE],
                [13, MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [MISSING_VALUE, 15, 14, 16],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [1, 3, 2, 4], self.get_state_normalization_parameters()
            )
        return dense

    def expected_tiled_next_state_features(self, normalize):
        # NOTE: this depends on lengths of possible next action
        # Feature order: 1, 3, 2, 4
        dense = np.array(
            [
                [11, MISSING_VALUE, 10, MISSING_VALUE],
                [11, MISSING_VALUE, 10, MISSING_VALUE],
                [13, MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [13, MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [MISSING_VALUE, 15, 14, 16],
                [MISSING_VALUE, 15, 14, 16],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [1, 3, 2, 4], self.get_state_normalization_parameters()
            )
        return dense

    def id_mapping_config(self):
        return {
            "a_mapping": rlt.IdMapping(ids=[20020, 20021]),
            "b_mapping": rlt.IdMapping(ids=[20031, 20030, 20032]),
            "c_mapping": rlt.IdMapping(ids=[20040, 20041, 20042, 20043, 20044]),
        }

    def setup_state_sequence_features(self, ws, id_list_field, id_score_list_field):
        # id_list
        id_list_lengths = np.array([3, 2, 0], dtype=np.int32)
        id_list_values_keys = np.array([2002, 2003, 2004, 2004, 2005], dtype=np.int64)
        id_list_values_values_lengths = np.array([2, 2, 1, 4, 1], dtype=np.int32)
        id_list_values_values_values = np.array(
            [20020, 20021, 20030, 20031, 20040, 20041, 20042, 20043, 20044, 20050],
            dtype=np.int64,
        )
        # id_score_list
        id_score_list_legnths = np.array([1, 2, 3], dtype=np.int32)
        id_score_list_values_keys = np.array(
            [1004, 1004, 1005, 1001, 1002, 1003], dtype=np.int64
        )
        id_score_list_values_values_lengths = np.array(
            [1, 4, 1, 1, 1, 1], dtype=np.int32
        )
        id_score_list_values_values_values_ids = np.array(
            [10040, 10041, 10042, 10043, 10044, 10050, 10010, 10020, 10030],
            dtype=np.int64,
        )
        id_score_list_values_values_values_scores = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32
        )
        ws.feed_blob(str(id_list_field.lengths()), id_list_lengths)
        ws.feed_blob(str(id_list_field["values"].keys()), id_list_values_keys)
        ws.feed_blob(
            str(id_list_field["values"]["values"].lengths()),
            id_list_values_values_lengths,
        )
        ws.feed_blob(
            str(id_list_field["values"]["values"]["values"]()),
            id_list_values_values_values,
        )

        ws.feed_blob(str(id_score_list_field.lengths()), id_score_list_legnths)
        ws.feed_blob(
            str(id_score_list_field["values"].keys()), id_score_list_values_keys
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"].lengths()),
            id_score_list_values_values_lengths,
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"]["values"]["keys"]()),
            id_score_list_values_values_values_ids,
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"]["values"]["values"]()),
            id_score_list_values_values_values_scores,
        )
        return {
            "id_list": {
                "lengths": id_list_lengths,
                "keys": id_list_values_keys,
                "values": {
                    "lengths": id_list_values_values_values,
                    "values": id_list_values_values_values,
                },
            },
            "id_score_list": {
                "lengths": id_score_list_legnths,
                "keys": id_score_list_values_keys,
                "values": {
                    "lengths": id_score_list_values_values_lengths,
                    "ids": id_score_list_values_values_values_ids,
                    "scores": id_score_list_values_values_values_scores,
                },
            },
        }

    def expected_state_sequence_features(self):
        return SequenceFeatures(
            id_only=IdOnlySequence(
                id_features=ABIdFeatures(
                    a_id=np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int64),
                    b_id=np.array([[2, 1], [0, 0], [0, 0]], dtype=np.int64),
                ),
                float_features=None,
            ),
            id_and_float=IdAndFloatSequence(
                id_features=CIdFeatures(
                    c_id=np.array([[1, 0, 0], [3, 4, 5], [0, 0, 0]], dtype=np.int64)
                ),
                float_features=np.array(
                    [
                        [[0.1], [0.0], [0.0]],
                        [[0.3], [0.4], [0.5]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
            float_only=FloatOnlySequence(
                id_features=None,
                float_features=np.array(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.7, 0.8, 0.9], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        )

    def setup_next_state_sequence_features(
        self, ws, id_list_field, id_score_list_field
    ):
        # id_list
        id_list_lengths = np.array([0, 3, 2], dtype=np.int32)
        id_list_values_keys = np.array([2002, 2003, 2004, 2004, 2005], dtype=np.int64)
        id_list_values_values_lengths = np.array([2, 2, 1, 4, 1], dtype=np.int32)
        id_list_values_values_values = np.array(
            [20020, 20021, 20030, 20031, 20040, 20041, 20042, 20043, 20044, 20050],
            dtype=np.int64,
        )
        # id_score_list
        id_score_list_legnths = np.array([3, 1, 2], dtype=np.int32)
        id_score_list_values_keys = np.array(
            [1001, 1002, 1003, 1004, 1004, 1005], dtype=np.int64
        )
        id_score_list_values_values_lengths = np.array(
            [1, 1, 1, 1, 4, 1], dtype=np.int32
        )
        id_score_list_values_values_values_ids = np.array(
            [10010, 10020, 10030, 10040, 10041, 10042, 10043, 10044, 10050],
            dtype=np.int64,
        )
        id_score_list_values_values_values_scores = np.array(
            [0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32
        )
        ws.feed_blob(str(id_list_field.lengths()), id_list_lengths)
        ws.feed_blob(str(id_list_field["values"].keys()), id_list_values_keys)
        ws.feed_blob(
            str(id_list_field["values"]["values"].lengths()),
            id_list_values_values_lengths,
        )
        ws.feed_blob(
            str(id_list_field["values"]["values"]["values"]()),
            id_list_values_values_values,
        )

        ws.feed_blob(str(id_score_list_field.lengths()), id_score_list_legnths)
        ws.feed_blob(
            str(id_score_list_field["values"].keys()), id_score_list_values_keys
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"].lengths()),
            id_score_list_values_values_lengths,
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"]["values"]["keys"]()),
            id_score_list_values_values_values_ids,
        )
        ws.feed_blob(
            str(id_score_list_field["values"]["values"]["values"]["values"]()),
            id_score_list_values_values_values_scores,
        )
        return {
            "id_list": {
                "lengths": id_list_lengths,
                "keys": id_list_values_keys,
                "values": {
                    "lengths": id_list_values_values_values,
                    "values": id_list_values_values_values,
                },
            },
            "id_score_list": {
                "lengths": id_score_list_legnths,
                "keys": id_score_list_values_keys,
                "values": {
                    "lengths": id_score_list_values_values_lengths,
                    "ids": id_score_list_values_values_values_ids,
                    "scores": id_score_list_values_values_values_scores,
                },
            },
        }

    def expected_next_state_sequence_features(self):
        return SequenceFeatures(
            id_only=IdOnlySequence(
                id_features=ABIdFeatures(
                    a_id=np.array([[0, 0], [1, 2], [0, 0]], dtype=np.int64),
                    b_id=np.array([[0, 0], [2, 1], [0, 0]], dtype=np.int64),
                ),
                float_features=None,
            ),
            id_and_float=IdAndFloatSequence(
                id_features=CIdFeatures(
                    c_id=np.array([[0, 0, 0], [1, 0, 0], [3, 4, 5]], dtype=np.int64)
                ),
                float_features=np.array(
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.1], [0.0], [0.0]],
                        [[0.3], [0.4], [0.5]],
                    ],
                    dtype=np.float32,
                ),
            ),
            float_only=FloatOnlySequence(
                id_features=None,
                float_features=np.array(
                    [
                        [[0.7, 0.8, 0.9], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        )

    def setup_action(self, ws, field):
        action = np.array([1, 0, 1], dtype=np.int64)
        ws.feed_blob(str(field()), action)
        return action

    def setup_next_action(self, ws, field):
        action = np.array([0, 1, 1], dtype=np.int64)
        ws.feed_blob(str(field()), action)
        return action

    def setup_possible_actions_mask(self, ws, field):
        lengths = np.array([2, 2, 2], dtype=np.int32)
        actions_mask = np.array([0, 1, 1, 1, 0, 1], dtype=np.int64)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"]()), actions_mask)
        return lengths, actions_mask

    def setup_possible_next_actions_mask(self, ws, field):
        lengths = np.array([2, 2, 2], dtype=np.int32)
        actions_mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"]()), actions_mask)
        return lengths, actions_mask

    def setup_action_features(self, ws, field):
        lengths = np.array([2, 4, 2], dtype=np.int32)
        keys = np.array([11, 12, 14, 11, 12, 13, 13, 12], dtype=np.int64)
        values = np.arange(20, 28).astype(np.float32)
        ws.feed_blob(str(field.lengths()), lengths)
        ws.feed_blob(str(field.keys()), keys)
        ws.feed_blob(str(field.values()), values)
        return lengths, keys, values

    def expected_action_features(self, normalize):
        # Feature order: 12, 11, 13
        dense = np.array(
            [[21, 20, MISSING_VALUE], [24, 23, 25], [27, MISSING_VALUE, 26]],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [12, 11, 13], self.get_action_normalization_parameters()
            )
        return dense

    def setup_next_action_features(self, ws, field):
        lengths = np.array([4, 2, 2], dtype=np.int32)
        keys = np.array([11, 12, 14, 13, 12, 13, 11, 13], dtype=np.int64)
        values = np.arange(30, 38).astype(np.float32)
        ws.feed_blob(str(field.lengths()), lengths)
        ws.feed_blob(str(field.keys()), keys)
        ws.feed_blob(str(field.values()), values)
        return lengths, keys, values

    def expected_next_action_features(self, normalize):
        # Feature order: 12, 11, 13
        dense = np.array(
            [[31, 30, 33], [34, MISSING_VALUE, 35], [MISSING_VALUE, 36, 37]],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [12, 11, 13], self.get_action_normalization_parameters()
            )
        return dense

    def setup_possible_actions_features(self, ws, field):
        lengths = np.array([2, 2, 2], dtype=np.int32)
        values_lengths = np.array([1, 0, 2, 3, 0, 0], dtype=np.int32)
        keys = np.array([11, 12, 14, 11, 13, 12], dtype=np.int64)
        values = np.arange(50, 56).astype(np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"].lengths()), values_lengths)
        ws.feed_blob(str(field["values"].keys()), keys)
        ws.feed_blob(str(field["values"].values()), values)
        return lengths, values_lengths, keys, values

    def expected_possible_actions_features(self, normalize):
        # Feature order: 12, 11, 13
        dense = np.array(
            [
                [MISSING_VALUE, 50, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [51, MISSING_VALUE, MISSING_VALUE],
                [55, 53, 54],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [12, 11, 13], self.get_action_normalization_parameters()
            )
        return dense

    def setup_possible_next_actions_features(self, ws, field):
        lengths = np.array([2, 2, 2], dtype=np.int32)
        values_lengths = np.array([1, 0, 2, 3, 0, 0], dtype=np.int32)
        keys = np.array([11, 12, 14, 11, 13, 12], dtype=np.int64)
        values = np.arange(40, 46).astype(np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"].lengths()), values_lengths)
        ws.feed_blob(str(field["values"].keys()), keys)
        ws.feed_blob(str(field["values"].values()), values)
        return lengths, values_lengths, keys, values

    def expected_possible_next_actions_features(self, normalize):
        # Feature order: 12, 11, 13
        dense = np.array(
            [
                [MISSING_VALUE, 40, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [41, MISSING_VALUE, MISSING_VALUE],
                [45, 43, 44],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [12, 11, 13], self.get_action_normalization_parameters()
            )
        return dense

    def setup_reward(self, ws, field):
        reward = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        ws.feed_blob(str(field()), reward)
        return reward

    def setup_not_terminal(self, ws, field):
        not_terminal = np.array([1, 1, 1], dtype=np.int32)
        ws.feed_blob(str(field()), not_terminal)
        return not_terminal

    def setup_step(self, ws, field):
        step = np.array([1, 2, 3], dtype=np.int32)
        ws.feed_blob(str(field()), step)
        return step

    def setup_time_diff(self, ws, field):
        time_diff = np.array([1, 1, 1], dtype=np.int32)
        ws.feed_blob(str(field()), time_diff)
        return time_diff

    def setup_time_since_first(self, ws, field):
        time_since_first = np.array([0, 0, 0], dtype=np.int32)
        ws.feed_blob(str(field()), time_since_first)
        return time_since_first

    def expected_time_since_first(self, normalize):
        dense = np.array([0, 0, 0], dtype=np.float).reshape(-1, 1)
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [0], {0: self.get_time_since_first_normalization_parameters()}
            )
        return dense

    def expected_next_time_since_first(self, normalize):
        dense = np.array([1, 1, 1], dtype=np.float).reshape(-1, 1)
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [0], {0: self.get_time_since_first_normalization_parameters()}
            )
        return dense

    def setup_mdp_id(self, ws, field):
        mdp_id = np.array(["1", "2", "3"], dtype=np.string_)
        ws.feed_blob(str(field()), mdp_id)
        return mdp_id

    def setup_seq_num(self, ws, field):
        seq_num = np.array([1, 1, 2], dtype=np.int32)
        ws.feed_blob(str(field()), seq_num)
        return seq_num

    def create_ws_and_net(self, extractor):
        net, init_net = extractor.create_net()
        ws = workspace.Workspace()
        ws.create_net(init_net)
        ws.run(init_net)
        for b in net.input_record().field_blobs():
            ws.create_blob(str(b))
        ws.create_net(net)
        return ws, net

    def check_create_net_spec(
        self,
        extractor,
        expected_input_record,
        expected_output_record,
        positional_match=True,
    ):
        net, init_net = extractor.create_net()
        # First, check that all outputs of init_net are used in net
        for b in init_net.external_outputs:
            self.assertTrue(net.is_external_input(b))
        # Second, check that input and output records are set
        input_record = net.input_record()
        output_record = net.output_record()
        self.assertIsNotNone(input_record)
        self.assertIsNotNone(output_record)
        # Third, check that the fields match what is expected
        self.assertEqual(
            set(expected_input_record.field_names()), set(input_record.field_names())
        )
        if positional_match:
            # Output must match positionally since it's used by exporting
            self.assertEqual(
                expected_output_record.field_names(), output_record.field_names()
            )
        else:
            # Training
            self.assertEqual(
                set(expected_output_record.field_names()),
                set(output_record.field_names()),
            )


class TestWorldModelFeatureExtractor(FeatureExtractorTestBase):
    SEQ_LEN = 3

    def expected_state_features(self, normalize):
        dense = super().expected_state_features(normalize)
        dense = dense.reshape((1, dense.shape[0], dense.shape[1]))
        return dense

    def expected_next_state_features(self, normalize):
        dense = super(
            TestWorldModelFeatureExtractor, self
        ).expected_next_state_features(normalize)
        dense = dense.reshape((1, dense.shape[0], dense.shape[1]))
        return dense

    def expected_action_features(self, normalize):
        dense = super().expected_action_features(normalize)
        dense = dense.reshape((1, dense.shape[0], dense.shape[1]))
        return dense

    def setup_state_features(self, ws, field):
        lengths = np.array([1], dtype=np.int32)
        values_lengths = np.array([3, 0, 5], dtype=np.int32)
        keys = np.array([2, 1, 9, 1, 2, 3, 4, 5], dtype=np.int64)
        values = np.arange(8).astype(np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"].lengths()), values_lengths)
        ws.feed_blob(str(field["values"].keys()), keys)
        ws.feed_blob(str(field["values"].values()), values)
        return lengths, values_lengths, keys, values

    def setup_next_state_features(self, ws, field):
        lengths = np.array([1], dtype=np.int32)
        values_lengths = np.array([2, 2, 4], dtype=np.int32)
        keys = np.array([2, 1, 9, 1, 2, 3, 4, 5], dtype=np.int64)
        values = np.arange(10, 18).astype(np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"].lengths()), values_lengths)
        ws.feed_blob(str(field["values"].keys()), keys)
        ws.feed_blob(str(field["values"].values()), values)
        return lengths, values_lengths, keys, values

    def setup_action_features(self, ws, field):
        lengths = np.array([1], dtype=np.int32)
        values_lengths = np.array([2, 4, 2], dtype=np.int32)
        keys = np.array([11, 12, 14, 11, 12, 13, 13, 12], dtype=np.int64)
        values = np.arange(20, 28).astype(np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"].lengths()), values_lengths)
        ws.feed_blob(str(field["values"].keys()), keys)
        ws.feed_blob(str(field["values"].values()), values)
        return lengths, values_lengths, keys, values

    def setup_action(self, ws, field):
        lengths = np.array([3], dtype=np.int32)
        action = np.array([1, 0, 1], dtype=np.int64)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"]()), action)
        return lengths, action

    def setup_reward(self, ws, field):
        lengths = np.array([3], dtype=np.int32)
        reward = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"]()), reward)
        return lengths, reward

    def setup_not_terminal(self, ws, field):
        lengths = np.array([3], dtype=np.int32)
        not_terminal = np.array([1, 1, 1], dtype=np.int32)
        ws.feed_blob(str(field["lengths"]()), lengths)
        ws.feed_blob(str(field["values"]()), not_terminal)
        return lengths, not_terminal

    def test_extract_parametric_action(self):
        self._test_extract_parametric_action(normalize=False)

    def test_extract_parametric_action_normalize(self):
        self._test_extract_parametric_action(normalize=True)

    def _test_extract_parametric_action(self, normalize):
        extractor = WorldModelFeatureExtractor(
            self.SEQ_LEN,
            self.get_state_normalization_parameters(),
            self.get_action_normalization_parameters(),
            normalize=normalize,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        self.setup_action_features(ws, input_record.action)
        _, reward = self.setup_reward(ws, input_record.reward)
        _, not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(1, 3), o.reward.numpy())
        npt.assert_array_equal(not_terminal.reshape(1, 3), o.not_terminal.numpy())
        npt.assert_allclose(
            self.expected_action_features(normalize),
            o.action.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.numpy(),
            rtol=1e-6,
        )

    def test_extract_discrete_action(self):
        self._test_extract_discrete_action(normalize=False)

    def test_extract_discrete_action_normalize(self):
        self._test_extract_discrete_action(normalize=True)

    def _test_extract_discrete_action(self, normalize):
        num_actions = 2
        extractor = WorldModelFeatureExtractor(
            self.SEQ_LEN,
            self.get_state_normalization_parameters(),
            discrete_action_names=["ACT1", "ACT2"],
            normalize=normalize,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        _, action = self.setup_action(ws, input_record.action)
        _, reward = self.setup_reward(ws, input_record.reward)
        _, not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(1, 3), o.reward.numpy())
        npt.assert_array_equal(not_terminal.reshape(1, 3), o.not_terminal.numpy())
        npt.assert_array_equal(
            action.reshape(-1, 3, 1) == np.arange(num_actions),
            o.action.float_features.numpy(),
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.numpy(),
            rtol=1e-6,
        )


class TestTrainingFeatureExtractor(FeatureExtractorTestBase):
    def setup_extra_data(self, ws, input_record):
        extra_data = rlt.ExtraData(
            action_probability=np.array([0.11, 0.21, 0.13], dtype=np.float32)
        )
        ws.feed_blob(
            str(input_record.action_probability()), extra_data.action_probability
        )
        return extra_data

    def pass_through_columns(self):
        return schema.Struct(
            ("reward", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("time_diff", schema.Scalar()),
            ("mdp_id", schema.Scalar()),
            ("sequence_number", schema.Scalar()),
            ("action_probability", schema.Scalar()),
        )

    def test_extract_max_q_discrete_action(self):
        self._test_extract_max_q_discrete_action(normalize=False)

    def test_extract_max_q_discrete_action_normalize(self):
        self._test_extract_max_q_discrete_action(normalize=True)

    def test_extract_max_q_discrete_action_time_since_first(self):
        self._test_extract_max_q_discrete_action(
            normalize=True, use_time_since_first=True
        )

    def test_extract_max_q_discrete_action_time_since_first_normalize(self):
        self._test_extract_max_q_discrete_action(
            normalize=True, use_time_since_first=True, normalize_time_since_first=True
        )

    def _test_extract_max_q_discrete_action(
        self, normalize, use_time_since_first=False, normalize_time_since_first=False
    ):
        num_actions = 2
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=True,
            normalize=normalize,
            max_num_actions=num_actions,
            multi_steps=3,
            use_time_since_first=use_time_since_first,
            time_since_first_normalization_parameters=self.get_time_since_first_normalization_parameters()
            if normalize_time_since_first
            else None,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        action = self.setup_action(ws, input_record.action)
        next_action = self.setup_next_action(ws, input_record.next_action)
        possible_actions_mask = self.setup_possible_actions_mask(
            ws, input_record.possible_actions_mask
        )
        possible_next_actions_mask = self.setup_possible_next_actions_mask(
            ws, input_record.possible_next_actions_mask
        )
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        if use_time_since_first:
            self.setup_time_since_first(ws, input_record.time_since_first)
        mdp_id = self.setup_mdp_id(ws, input_record.mdp_id)
        sequence_number = self.setup_seq_num(ws, input_record.sequence_number)
        step = self.setup_step(ws, input_record.step)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        e = res.extras
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        if use_time_since_first:
            npt.assert_allclose(
                self.expected_time_since_first(normalize_time_since_first),
                o.state.time_since_first.numpy(),
                rtol=1e-6,
            )
            npt.assert_allclose(
                self.expected_next_time_since_first(normalize_time_since_first),
                o.next_state.time_since_first.numpy(),
                rtol=1e-6,
            )
        else:
            self.assertIsNone(o.state.time_since_first)
            self.assertIsNone(o.next_state.time_since_first)
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(step.reshape(-1, 1), o.step.numpy())
        npt.assert_array_equal(
            sequence_number.reshape(-1, 1), e.sequence_number.numpy()
        )
        npt.assert_array_equal(mdp_id.reshape(-1, 1), e.mdp_id)
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_array_equal(
            action.reshape(-1, 1) == np.arange(num_actions), o.action.numpy()
        )
        npt.assert_array_equal(
            next_action.reshape(-1, 1) == np.arange(num_actions), o.next_action.numpy()
        )
        npt.assert_array_equal(
            possible_actions_mask[1], o.possible_actions_mask.numpy().flatten()
        )
        npt.assert_array_equal(
            possible_next_actions_mask[1],
            o.possible_next_actions_mask.numpy().flatten(),
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.float_features.numpy(),
            rtol=1e-6,
        )

    def test_extract_max_q_discrete_action_with_sequence(self):
        normalize = True
        num_actions = 2
        model_feature_config = rlt.ModelFeatureConfig(
            id_mapping_config=self.id_mapping_config(),
            sequence_features_type=SequenceFeatures,
            float_feature_infos=[],
        )
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=True,
            normalize=normalize,
            max_num_actions=num_actions,
            model_feature_config=model_feature_config,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        self.setup_state_sequence_features(
            ws,
            input_record.state_id_list_features,
            input_record.state_id_score_list_features,
        )
        self.setup_next_state_sequence_features(
            ws,
            input_record.next_state_id_list_features,
            input_record.next_state_id_score_list_features,
        )
        action = self.setup_action(ws, input_record.action)
        next_action = self.setup_next_action(ws, input_record.next_action)
        possible_actions_mask = self.setup_possible_actions_mask(
            ws, input_record.possible_actions_mask
        )
        possible_next_actions_mask = self.setup_possible_next_actions_mask(
            ws, input_record.possible_next_actions_mask
        )
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        mdp_id = self.setup_mdp_id(ws, input_record.mdp_id)
        sequence_number = self.setup_seq_num(ws, input_record.sequence_number)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        e = res.extras
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(
            sequence_number.reshape(-1, 1), e.sequence_number.numpy()
        )
        npt.assert_array_equal(mdp_id.reshape(-1, 1), e.mdp_id)
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_array_equal(
            action.reshape(-1, 1) == np.arange(num_actions), o.action.numpy()
        )
        npt.assert_array_equal(
            next_action.reshape(-1, 1) == np.arange(num_actions), o.next_action.numpy()
        )
        npt.assert_array_equal(
            possible_actions_mask[1], o.possible_actions_mask.numpy().flatten()
        )
        npt.assert_array_equal(
            possible_next_actions_mask[1],
            o.possible_next_actions_mask.numpy().flatten(),
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.float_features.numpy(),
            rtol=1e-6,
        )

        # Check state sequence features
        expected_state_sequence_features = self.expected_state_sequence_features()

        id_only = o.state.sequence_features.id_only
        expected_id_only = expected_state_sequence_features.id_only
        self.assertEqual(expected_id_only.float_features, id_only.float_features)
        npt.assert_array_equal(
            expected_id_only.id_features.a_id, id_only.id_features.a_id
        )
        npt.assert_array_equal(
            expected_id_only.id_features.b_id, id_only.id_features.b_id
        )

        id_and_float = o.state.sequence_features.id_and_float
        expected_id_and_float = expected_state_sequence_features.id_and_float
        npt.assert_array_equal(
            expected_id_and_float.float_features, id_and_float.float_features
        )
        npt.assert_array_equal(
            expected_id_and_float.id_features.c_id, id_and_float.id_features.c_id
        )

        float_only = o.state.sequence_features.float_only
        expected_float_only = expected_state_sequence_features.float_only
        npt.assert_array_equal(
            expected_float_only.float_features, float_only.float_features
        )
        self.assertEqual(expected_float_only.id_features, float_only.id_features)

        # Check next state sequence features
        expected_next_state_sequence_features = (
            self.expected_next_state_sequence_features()
        )

        id_only = o.next_state.sequence_features.id_only
        expected_id_only = expected_next_state_sequence_features.id_only
        self.assertEqual(expected_id_only.float_features, id_only.float_features)
        npt.assert_array_equal(
            expected_id_only.id_features.a_id, id_only.id_features.a_id
        )
        npt.assert_array_equal(
            expected_id_only.id_features.b_id, id_only.id_features.b_id
        )

        id_and_float = o.next_state.sequence_features.id_and_float
        expected_id_and_float = expected_next_state_sequence_features.id_and_float
        npt.assert_array_equal(
            expected_id_and_float.float_features, id_and_float.float_features
        )
        npt.assert_array_equal(
            expected_id_and_float.id_features.c_id, id_and_float.id_features.c_id
        )

        float_only = o.next_state.sequence_features.float_only
        expected_float_only = expected_next_state_sequence_features.float_only
        npt.assert_array_equal(
            expected_float_only.float_features, float_only.float_features
        )
        self.assertEqual(expected_float_only.id_features, float_only.id_features)

    def test_extract_sarsa_discrete_action(self):
        self._test_extract_sarsa_discrete_action(normalize=False)

    def test_extract_sarsa_discrete_action_normalize(self):
        self._test_extract_sarsa_discrete_action(normalize=True)

    def _test_extract_sarsa_discrete_action(self, normalize):
        num_actions = 2
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=False,
            normalize=normalize,
            max_num_actions=num_actions,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        action = self.setup_action(ws, input_record.action)
        next_action = self.setup_next_action(ws, input_record.next_action)
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        mdp_id = self.setup_mdp_id(ws, input_record.mdp_id)
        sequence_number = self.setup_seq_num(ws, input_record.sequence_number)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        e = res.extras
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(
            sequence_number.reshape(-1, 1), e.sequence_number.numpy()
        )
        npt.assert_array_equal(mdp_id.reshape(-1, 1), e.mdp_id)
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_array_equal(
            action.reshape(-1, 1) == np.arange(num_actions), o.action.numpy()
        )
        npt.assert_array_equal(
            next_action.reshape(-1, 1) == np.arange(num_actions), o.next_action.numpy()
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.float_features.numpy(),
            rtol=1e-6,
        )

    def test_extract_max_q_parametric_action(self):
        self._test_extract_max_q_parametric_action(normalize=False)

    def test_extract_max_q_parametric_action_normalize(self):
        self._test_extract_max_q_parametric_action(normalize=True)

    def _test_extract_max_q_parametric_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            include_possible_actions=True,
            normalize=normalize,
            max_num_actions=2,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        self.setup_action_features(ws, input_record.action)
        self.setup_next_action_features(ws, input_record.next_action)
        self.setup_possible_actions_features(ws, input_record.possible_actions)
        possible_actions_mask = self.setup_possible_actions_mask(
            ws, input_record.possible_actions_mask
        )
        self.setup_possible_next_actions_features(
            ws, input_record.possible_next_actions
        )
        possible_next_actions_mask = self.setup_possible_next_actions_mask(
            ws, input_record.possible_next_actions_mask
        )
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        mdp_id = self.setup_mdp_id(ws, input_record.mdp_id)
        sequence_number = self.setup_seq_num(ws, input_record.sequence_number)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        e = res.extras
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(
            sequence_number.reshape(-1, 1), e.sequence_number.numpy()
        )
        npt.assert_array_equal(mdp_id.reshape(-1, 1), e.mdp_id)
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_allclose(
            self.expected_action_features(normalize),
            o.action.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_possible_actions_features(normalize),
            o.possible_actions.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_array_equal(
            possible_actions_mask[1], o.possible_actions_mask.numpy().flatten()
        )
        npt.assert_allclose(
            self.expected_possible_next_actions_features(normalize),
            o.possible_next_actions.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_array_equal(
            possible_next_actions_mask[1],
            o.possible_next_actions_mask.numpy().flatten(),
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_tiled_next_state_features(normalize),
            o.tiled_next_state.float_features.numpy(),
            rtol=1e-6,
        )

    def test_extract_sarsa_parametric_action(self):
        self._test_extract_sarsa_parametric_action(normalize=False)

    def test_extract_sarsa_parametric_action_normalize(self):
        self._test_extract_sarsa_parametric_action(normalize=True)

    def _test_extract_sarsa_parametric_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            include_possible_actions=False,
            normalize=normalize,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        self.setup_action_features(ws, input_record.action)
        self.setup_next_action_features(ws, input_record.next_action)
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        mdp_id = self.setup_mdp_id(ws, input_record.mdp_id)
        sequence_number = self.setup_seq_num(ws, input_record.sequence_number)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        o = res.training_input
        e = res.extras
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(
            sequence_number.reshape(-1, 1), e.sequence_number.numpy()
        )
        npt.assert_array_equal(mdp_id.reshape(-1, 1), e.mdp_id)
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_allclose(
            self.expected_action_features(normalize),
            o.action.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_action_features(normalize),
            o.next_action.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            o.state.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_next_state_features(normalize),
            o.next_state.float_features.numpy(),
            rtol=1e-6,
        )

    def test_create_net_max_q_discrete_action(self):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=True,
            max_num_actions=2,
        )
        expected_input_record = (
            schema.Struct(
                ("state_features", map_schema()),
                ("next_state_features", map_schema()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
                ("possible_actions_mask", schema.List(schema.Scalar())),
                ("possible_next_actions_mask", schema.List(schema.Scalar())),
            )
            + self.pass_through_columns()
        )
        expected_output_record = (
            schema.Struct(
                ("state_features", schema.Scalar()),
                ("next_state_features", schema.Scalar()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
                ("possible_actions_mask", schema.Scalar()),
                ("possible_next_actions_mask", schema.Scalar()),
            )
            + self.pass_through_columns()
        )
        self.check_create_net_spec(
            extractor,
            expected_input_record,
            expected_output_record,
            positional_match=False,
        )

    def test_create_net_sarsa_discrete_action(self):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=False,
            max_num_actions=2,
        )
        expected_input_record = (
            schema.Struct(
                ("state_features", map_schema()),
                ("next_state_features", map_schema()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
            )
            + self.pass_through_columns()
        )
        expected_output_record = (
            schema.Struct(
                ("state_features", schema.Scalar()),
                ("next_state_features", schema.Scalar()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
            )
            + self.pass_through_columns()
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )

    def test_create_net_max_q_parametric_action(self):
        self._test_create_net_max_q_parametric_action(normalize=False)

    def test_create_net_max_q_parametric_action_normalize(self):
        self._test_create_net_max_q_parametric_action(normalize=True)

    def _test_create_net_max_q_parametric_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            include_possible_actions=True,
            normalize=normalize,
            max_num_actions=2,
        )
        expected_input_record = (
            schema.Struct(
                ("state_features", map_schema()),
                ("next_state_features", map_schema()),
                ("action", map_schema()),
                ("next_action", map_schema()),
                ("possible_actions", schema.List(map_schema())),
                ("possible_actions_mask", schema.List(schema.Scalar())),
                ("possible_next_actions", schema.List(map_schema())),
                ("possible_next_actions_mask", schema.List(schema.Scalar())),
            )
            + self.pass_through_columns()
        )
        expected_output_record = (
            schema.Struct(
                ("state_features", schema.Scalar()),
                ("next_state_features", schema.Scalar()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
                ("possible_actions_mask", schema.Scalar()),
                ("possible_next_actions_mask", schema.Scalar()),
                ("possible_actions", schema.Scalar()),
                ("possible_next_actions", schema.Scalar()),
            )
            + self.pass_through_columns()
        )
        self.check_create_net_spec(
            extractor,
            expected_input_record,
            expected_output_record,
            positional_match=False,
        )

    def test_create_net_sarsa_parametric_action(self):
        self._test_create_net_sarsa_parametric_action(normalize=False)

    def test_create_net_sarsa_parametric_action_normalize(self):
        self._test_create_net_sarsa_parametric_action(normalize=True)

    def _test_create_net_sarsa_parametric_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            include_possible_actions=False,
            normalize=normalize,
            max_num_actions=2,
        )
        expected_input_record = (
            schema.Struct(
                ("state_features", map_schema()),
                ("next_state_features", map_schema()),
                ("action", map_schema()),
                ("next_action", map_schema()),
            )
            + self.pass_through_columns()
        )
        expected_output_record = (
            schema.Struct(
                ("state_features", schema.Scalar()),
                ("next_state_features", schema.Scalar()),
                ("action", schema.Scalar()),
                ("next_action", schema.Scalar()),
            )
            + self.pass_through_columns()
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )


class TestPredictorFeatureExtractor(FeatureExtractorTestBase):
    def setup_float_features(self, ws, field):
        lengths = np.array([3 + 2, 0 + 4, 5 + 2], dtype=np.int32)
        keys = np.array(
            [2, 11, 12, 1, 14, 11, 12, 13, 9, 1, 13, 12, 2, 3, 4, 5], dtype=np.int64
        )
        values = np.array(
            [0, 20, 21, 1, 22, 23, 24, 25, 2, 3, 26, 27, 4, 5, 6, 7], dtype=np.float32
        )
        ws.feed_blob(str(field.lengths()), lengths)
        ws.feed_blob(str(field.keys()), keys)
        ws.feed_blob(str(field.values()), values)
        return lengths, keys, values

    def expected_state_features(self, normalize):
        # Feature order: 1, 3, 2, 4
        dense = np.array(
            [
                [1, MISSING_VALUE, 0, MISSING_VALUE],
                [MISSING_VALUE, MISSING_VALUE, MISSING_VALUE, MISSING_VALUE],
                [3, 5, 4, 6],
            ],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [1, 3, 2, 4], self.get_state_normalization_parameters()
            )
        return dense

    def expected_action_features(self, normalize):
        # Feature order: 12, 11, 13
        dense = np.array(
            [[21, 20, MISSING_VALUE], [24, 23, 25], [27, MISSING_VALUE, 26]],
            dtype=np.float32,
        )
        if normalize:
            dense = NumpyFeatureProcessor.preprocess_array(
                dense, [12, 11, 13], self.get_action_normalization_parameters()
            )
        return dense

    def test_extract_no_action(self):
        self._test_extract_no_action(normalize=False)

    def test_extract_no_action_normalize(self):
        self._test_extract_no_action(normalize=True)

    def test_extract_no_action_time_since_first(self):
        self._test_extract_no_action(normalize=True, use_time_since_first=True)

    def test_extract_no_action_time_since_first_normalize(self):
        self._test_extract_no_action(
            normalize=True, use_time_since_first=True, normalize_time_since_first=True
        )

    def _test_extract_no_action(
        self, normalize, use_time_since_first=False, normalize_time_since_first=False
    ):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            normalize=normalize,
            use_time_since_first=use_time_since_first,
            time_since_first_normalization_parameters=self.get_time_since_first_normalization_parameters()
            if normalize_time_since_first
            else None,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_float_features(ws, input_record.float_features)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        npt.assert_allclose(
            self.expected_state_features(normalize),
            res.state.float_features.numpy(),
            rtol=1e-6,
        )
        if use_time_since_first:
            npt.assert_allclose(
                self.expected_time_since_first(normalize_time_since_first),
                res.state.time_since_first.numpy(),
                rtol=1e-6,
            )
        else:
            self.assertIsNone(res.state.time_since_first)

    def test_extract_with_sequence(self):
        model_feature_config = rlt.ModelFeatureConfig(
            id_mapping_config=self.id_mapping_config(),
            sequence_features_type=SequenceFeatures,
            float_feature_infos=[],
        )
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            normalize=True,
            model_feature_config=model_feature_config,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_state_features(ws, input_record.float_features)
        self.setup_state_sequence_features(
            ws, input_record.id_list_features, input_record.id_score_list_features
        )
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        npt.assert_allclose(
            super().expected_state_features(normalize=True),
            res.state.float_features.numpy(),
            rtol=1e-6,
        )
        # Check state sequence features
        expected_state_sequence_features = self.expected_state_sequence_features()

        id_only = res.state.sequence_features.id_only
        expected_id_only = expected_state_sequence_features.id_only
        self.assertEqual(expected_id_only.float_features, id_only.float_features)
        npt.assert_array_equal(
            expected_id_only.id_features.a_id, id_only.id_features.a_id
        )
        npt.assert_array_equal(
            expected_id_only.id_features.b_id, id_only.id_features.b_id
        )

        id_and_float = res.state.sequence_features.id_and_float
        expected_id_and_float = expected_state_sequence_features.id_and_float
        npt.assert_array_equal(
            expected_id_and_float.float_features, id_and_float.float_features
        )
        npt.assert_array_equal(
            expected_id_and_float.id_features.c_id, id_and_float.id_features.c_id
        )

        float_only = res.state.sequence_features.float_only
        expected_float_only = expected_state_sequence_features.float_only
        npt.assert_array_equal(
            expected_float_only.float_features, float_only.float_features
        )
        self.assertEqual(expected_float_only.id_features, float_only.id_features)

    def test_extract_parametric_action(self):
        self._test_extract_parametric_action(normalize=False)

    def test_extract_parametric_action_normalize(self):
        self._test_extract_parametric_action(normalize=True)

    def _test_extract_parametric_action(self, normalize):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            normalize=normalize,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_float_features(ws, input_record.float_features)
        # Run
        ws.run(net)
        res = extractor.extract(ws, net.output_record())
        npt.assert_allclose(
            self.expected_action_features(normalize),
            res.action.float_features.numpy(),
            rtol=1e-6,
        )
        npt.assert_allclose(
            self.expected_state_features(normalize),
            res.state.float_features.numpy(),
            rtol=1e-6,
        )

    def test_create_net_sarsa_no_action(self):
        self._test_create_net_sarsa_no_action(normalize=False)

    def test_create_net_sarsa_no_action_normalize(self):
        self._test_create_net_sarsa_no_action(normalize=True)

    def _test_create_net_sarsa_no_action(self, normalize):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            normalize=normalize,
        )
        expected_input_record = schema.Struct(("float_features", map_schema()))
        expected_output_record = schema.Struct(
            ("state:float_features", schema.Scalar())
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )

    def test_create_net_sarsa_no_action_with_sequence(self):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            normalize=True,
            model_feature_config=rlt.ModelFeatureConfig(
                id_mapping_config=self.id_mapping_config(),
                sequence_features_type=SequenceFeatures,
                float_feature_infos=[],
            ),
        )
        expected_input_record = schema.Struct(
            ("float_features", map_schema()),
            ("id_list_features", id_list_schema()),
            ("id_score_list_features", id_score_list_schema()),
        )
        expected_output_record = schema.Struct(
            ("state:float_features", schema.Scalar()),
            ("state:sequence_features:id_only:id_features:a_id", schema.Scalar()),
            ("state:sequence_features:id_only:id_features:b_id", schema.Scalar()),
            ("state:sequence_features:id_and_float:id_features:c_id", schema.Scalar()),
            ("state:sequence_features:id_and_float:float_features", schema.Scalar()),
            ("state:sequence_features:float_only:float_features", schema.Scalar()),
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )

    def test_create_net_parametric_action(self):
        self._test_create_net_parametric_action(normalize=False)

    def test_create_net_parametric_action_normalize(self):
        self._test_create_net_parametric_action(normalize=True)

    def _test_create_net_parametric_action(self, normalize):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            action_normalization_parameters=self.get_action_normalization_parameters(),
            normalize=normalize,
        )
        expected_input_record = schema.Struct(("float_features", map_schema()))
        expected_output_record = schema.Struct(
            ("state:float_features", schema.Scalar()), ("action", schema.Scalar())
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )
