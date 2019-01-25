#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
from caffe2.python import schema, workspace
from ml.rl import types as rlt
from ml.rl.preprocessing.feature_extractor import (
    PredictorFeatureExtractor,
    TrainingFeatureExtractor,
    map_schema,
)
from ml.rl.preprocessing.identify_types import CONTINUOUS, PROBABILITY
from ml.rl.preprocessing.normalization import MISSING_VALUE, NormalizationParameters
from ml.rl.test.utils import NumpyFeatureProcessor


class FeatureExtractorTestBase(unittest.TestCase):
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
        self, extractor, expected_input_record, expected_output_record
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
        self.assertEqual(
            set(expected_output_record.field_names()), set(output_record.field_names())
        )


class TestTrainingFeatureExtractor(FeatureExtractorTestBase):
    def create_extra_input_record(self, net):
        return net.input_record() + schema.NewRecord(
            net,
            schema.Struct(
                ("reward", schema.Scalar()),
                ("action_probability", schema.Scalar()),
                ("step", schema.Scalar()),
            ),
        )

    def setup_extra_data(self, ws, input_record):
        extra_data = rlt.ExtraData(
            action_probability=np.array([0.11, 0.21, 0.13], dtype=np.float32)
        )
        ws.feed_blob(
            str(input_record.action_probability()), extra_data.action_probability
        )
        return extra_data

    def test_extract_max_q_discrete_action(self):
        self._test_extract_max_q_discrete_action(normalize=False)

    def test_extract_max_q_discrete_action_normalize(self):
        self._test_extract_max_q_discrete_action(normalize=True)

    def _test_extract_max_q_discrete_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=True,
            normalize=normalize,
            max_num_actions=2,
            multi_steps=3,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = self.create_extra_input_record(net)
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
        step = self.setup_step(ws, input_record.step)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, input_record, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(step.reshape(-1, 1), o.step.numpy())
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_array_equal(action, o.action.numpy())
        npt.assert_array_equal(next_action, o.next_action.numpy())
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

    def test_extract_sarsa_discrete_action(self):
        self._test_extract_sarsa_discrete_action(normalize=False)

    def test_extract_sarsa_discrete_action_normalize(self):
        self._test_extract_sarsa_discrete_action(normalize=True)

    def _test_extract_sarsa_discrete_action(self, normalize):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=False,
            normalize=normalize,
            max_num_actions=2,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = self.create_extra_input_record(net)
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        action = self.setup_action(ws, input_record.action)
        next_action = self.setup_next_action(ws, input_record.next_action)
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, input_record, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
        npt.assert_array_equal(
            extra_data.action_probability.reshape(-1, 1),
            res.extras.action_probability.numpy(),
        )
        npt.assert_array_equal(action, o.action.numpy())
        npt.assert_array_equal(next_action, o.next_action.numpy())
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
        input_record = self.create_extra_input_record(net)
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
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, input_record, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
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
        input_record = self.create_extra_input_record(net)
        self.setup_state_features(ws, input_record.state_features)
        self.setup_next_state_features(ws, input_record.next_state_features)
        self.setup_action_features(ws, input_record.action)
        self.setup_next_action_features(ws, input_record.next_action)
        reward = self.setup_reward(ws, input_record.reward)
        not_terminal = self.setup_not_terminal(ws, input_record.not_terminal)
        time_diff = self.setup_time_diff(ws, input_record.time_diff)
        extra_data = self.setup_extra_data(ws, input_record)
        # Run
        ws.run(net)
        res = extractor.extract(ws, input_record, net.output_record())
        o = res.training_input
        npt.assert_array_equal(reward.reshape(-1, 1), o.reward.numpy())
        npt.assert_array_equal(time_diff.reshape(-1, 1), o.time_diff.numpy())
        npt.assert_array_equal(not_terminal.reshape(-1, 1), o.not_terminal.numpy())
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
        expected_input_record = schema.Struct(
            ("state_features", map_schema()),
            ("next_state_features", map_schema()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("possible_actions_mask", schema.List(schema.Scalar())),
            ("possible_next_actions_mask", schema.List(schema.Scalar())),
            ("time_diff", schema.Scalar()),
        )
        expected_output_record = schema.Struct(
            ("state_features", schema.Scalar()),
            ("next_state_features", schema.Scalar()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("possible_actions_mask", schema.Scalar()),
            ("possible_next_actions_mask", schema.Scalar()),
            ("time_diff", schema.Scalar()),
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )

    def test_create_net_sarsa_discrete_action(self):
        extractor = TrainingFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            include_possible_actions=False,
            max_num_actions=2,
        )
        expected_input_record = schema.Struct(
            ("state_features", map_schema()),
            ("next_state_features", map_schema()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("time_diff", schema.Scalar()),
        )
        expected_output_record = schema.Struct(
            ("state_features", schema.Scalar()),
            ("next_state_features", schema.Scalar()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("time_diff", schema.Scalar()),
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
        expected_input_record = schema.Struct(
            ("state_features", map_schema()),
            ("next_state_features", map_schema()),
            ("action", map_schema()),
            ("next_action", map_schema()),
            ("not_terminal", schema.Scalar()),
            ("possible_actions", schema.List(map_schema())),
            ("possible_actions_mask", schema.List(schema.Scalar())),
            ("possible_next_actions", schema.List(map_schema())),
            ("possible_next_actions_mask", schema.List(schema.Scalar())),
            ("time_diff", schema.Scalar()),
        )
        expected_output_record = schema.Struct(
            ("state_features", schema.Scalar()),
            ("next_state_features", schema.Scalar()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("possible_actions", schema.Scalar()),
            ("possible_actions_mask", schema.Scalar()),
            ("possible_next_actions", schema.Scalar()),
            ("possible_next_actions_mask", schema.Scalar()),
            ("time_diff", schema.Scalar()),
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
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
        expected_input_record = schema.Struct(
            ("state_features", map_schema()),
            ("next_state_features", map_schema()),
            ("action", map_schema()),
            ("next_action", map_schema()),
            ("not_terminal", schema.Scalar()),
            ("time_diff", schema.Scalar()),
        )
        expected_output_record = schema.Struct(
            ("state_features", schema.Scalar()),
            ("next_state_features", schema.Scalar()),
            ("action", schema.Scalar()),
            ("next_action", schema.Scalar()),
            ("not_terminal", schema.Scalar()),
            ("time_diff", schema.Scalar()),
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
        # values = np.arange(8).astype(np.float32)
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

    def _test_extract_no_action(self, normalize):
        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=self.get_state_normalization_parameters(),
            normalize=normalize,
        )
        # Setup
        ws, net = self.create_ws_and_net(extractor)
        input_record = net.input_record()
        self.setup_float_features(ws, input_record.float_features)
        # Run
        ws.run(net)
        res = extractor.extract(ws, input_record, net.output_record())
        npt.assert_allclose(
            self.expected_state_features(normalize),
            res.state.float_features.numpy(),
            rtol=1e-6,
        )

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
        res = extractor.extract(ws, input_record, net.output_record())
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
        expected_output_record = schema.Struct(("state", schema.Scalar()))
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
            ("state", schema.Scalar()), ("action", schema.Scalar())
        )
        self.check_create_net_spec(
            extractor, expected_input_record, expected_output_record
        )
