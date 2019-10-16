#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
from caffe2.python import core, schema, workspace
from ml.rl.models.output_transformer import ActorOutputTransformer


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
