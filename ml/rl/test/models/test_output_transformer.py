#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
from caffe2.python import core, schema, workspace
from ml.rl.models.output_transformer import (
    ActorOutputTransformer,
    DiscreteActionOutputTransformer,
    ParametricActionOutputTransformer,
)


class DiscreteActionOutputTransformerTest(unittest.TestCase):
    def test_create_net(self):
        q_value_blob = core.BlobReference("q_values")
        N = 10
        # NOTE: We add `b` prefix here to match the return type of FetchBlob
        actions = [b"yes", b"no"]
        q_values = np.random.randn(N, len(actions)).astype(np.float32)
        workspace.FeedBlob(q_value_blob, q_values)
        ot = DiscreteActionOutputTransformer(actions)
        output_record = schema.Struct(("q_values", schema.Scalar(blob=q_value_blob)))
        nets = ot.create_net(output_record)
        workspace.RunNetOnce(nets.init_net)
        workspace.RunNetOnce(nets.net)

        external_outputs = {str(b) for b in nets.net.external_outputs}

        def fetch_blob(b):
            self.assertIn(b, external_outputs)
            return workspace.FetchBlob(b)

        feature_lengths = fetch_blob(
            "output/string_weighted_multi_categorical_features.lengths"
        )
        feature_keys = fetch_blob(
            "output/string_weighted_multi_categorical_features.keys"
        )
        values_lengths = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        values_keys = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.keys"
        )
        values_values = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.values"
        )
        action_lengths = fetch_blob("output/string_single_categorical_features.lengths")
        action_keys = fetch_blob("output/string_single_categorical_features.keys")
        action_values = fetch_blob("output/string_single_categorical_features.values")

        npt.assert_array_equal(np.ones(N, dtype=np.int32), feature_lengths)
        npt.assert_array_equal(np.zeros(N, dtype=np.int64), feature_keys)
        npt.assert_array_equal([len(actions)] * N, values_lengths)
        npt.assert_array_equal(np.array(actions * N, dtype=np.object), values_keys)
        npt.assert_array_equal(q_values.reshape(-1), values_values)
        npt.assert_array_equal([len(actions)] * N, action_lengths)
        npt.assert_array_equal(list(range(len(actions))) * N, action_keys)

        # We can only assert max-Q policy
        max_q_actions = action_values.reshape(-1, len(actions))[::, 0]
        npt.assert_array_equal(
            [actions[i] for i in np.argmax(q_values, axis=1)], max_q_actions
        )


class ParametricActionOutputTransformerTest(unittest.TestCase):
    def test_create_net(self):
        q_value_blob = core.BlobReference("q_values")
        N = 10
        # NOTE: We add `b` prefix here to match the return type of FetchBlob
        actions = [b"Q"]
        q_values = np.random.randn(N, len(actions)).astype(np.float32)
        workspace.FeedBlob(q_value_blob, q_values)
        ot = ParametricActionOutputTransformer()
        output_record = schema.Struct(("q_value", schema.Scalar(blob=q_value_blob)))
        nets = ot.create_net(output_record)
        workspace.RunNetOnce(nets.init_net)
        workspace.RunNetOnce(nets.net)

        external_outputs = {str(b) for b in nets.net.external_outputs}

        def fetch_blob(b):
            self.assertIn(b, external_outputs)
            return workspace.FetchBlob(b)

        feature_lengths = fetch_blob(
            "output/string_weighted_multi_categorical_features.lengths"
        )
        feature_keys = fetch_blob(
            "output/string_weighted_multi_categorical_features.keys"
        )
        values_lengths = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        values_keys = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.keys"
        )
        values_values = fetch_blob(
            "output/string_weighted_multi_categorical_features.values.values"
        )

        npt.assert_array_equal(np.ones(N, dtype=np.int32), feature_lengths)
        npt.assert_array_equal(np.zeros(N, dtype=np.int64), feature_keys)
        npt.assert_array_equal([len(actions)] * N, values_lengths)
        npt.assert_array_equal(np.array(actions * N, dtype=np.object), values_keys)
        npt.assert_array_equal(q_values.reshape(-1), values_values)


class ActorOutputTransformerTest(unittest.TestCase):
    def test_create_net(self):
        action_blob = core.BlobReference("action")

        N = 10
        action_feature_ids = [100, 300, 200]
        serving_max_scale = np.array(action_feature_ids) / 100.0
        serving_min_scale = np.zeros(len(action_feature_ids)) - 5.0
        actions = np.random.randn(N, len(action_feature_ids)).astype(np.float32)
        workspace.FeedBlob(action_blob, actions)
        ot = ActorOutputTransformer(
            action_feature_ids, serving_max_scale, serving_min_scale
        )
        output_record = schema.Struct(("action", schema.Scalar(blob=action_blob)))
        nets = ot.create_net(output_record)
        workspace.RunNetOnce(nets.init_net)
        workspace.RunNetOnce(nets.net)

        external_outputs = {str(b) for b in nets.net.external_outputs}

        def fetch_blob(b):
            self.assertIn(b, external_outputs)
            return workspace.FetchBlob(b)

        lengths = fetch_blob("output/float_features.lengths")
        keys = fetch_blob("output/float_features.keys")
        values = fetch_blob("output/float_features.values")

        scaled_actions = (actions + np.ones(len(action_feature_ids)) - 1e-6) / (
            (1 - 1e-6) * 2
        ) * (serving_max_scale - serving_min_scale) + serving_min_scale

        npt.assert_array_equal([len(action_feature_ids)] * N, lengths)
        npt.assert_array_equal(action_feature_ids * N, keys)
        npt.assert_array_almost_equal(scaled_actions.reshape(-1), values, decimal=3)
