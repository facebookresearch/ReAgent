#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

# pyre-fixme[21]: Could not find module `gym`.
import gym

# pyre-fixme[21]: Could not find module `numpy.testing`.
import numpy.testing as npt
import torch
import torch.nn.functional as F
from reagent.gym.envs import Gym


try:
    from reagent.gym.envs import RecSim

    HAS_RECSIM = True
except ModuleNotFoundError:
    HAS_RECSIM = False


class TestMakeDefaultObsPreprocessor(unittest.TestCase):
    def test_box(self):
        env = Gym(env_name="CartPole-v0")
        obs_preprocessor = env.get_obs_preprocessor()
        obs = env.reset()
        state = obs_preprocessor(obs)
        self.assertTrue(state.has_float_features_only)
        self.assertEqual(state.float_features.shape, (1, obs.shape[0]))
        self.assertEqual(state.float_features.dtype, torch.float32)
        self.assertEqual(state.float_features.device, torch.device("cpu"))
        npt.assert_array_almost_equal(obs, state.float_features.squeeze(0))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_box_cuda(self):
        env = Gym(env_name="CartPole-v0")
        device = torch.device("cuda")
        obs_preprocessor = env.get_obs_preprocessor(device=device)
        obs = env.reset()
        state = obs_preprocessor(obs)
        self.assertTrue(state.has_float_features_only)
        self.assertEqual(state.float_features.shape, (1, obs.shape[0]))
        self.assertEqual(state.float_features.dtype, torch.float32)
        # `device` doesn't have index. So we need this.
        x = torch.zeros(1, device=device)
        self.assertEqual(state.float_features.device, x.device)
        npt.assert_array_almost_equal(obs, state.float_features.cpu().squeeze(0))

    @unittest.skipIf(not HAS_RECSIM, "Recsim is not installed")
    def test_recsim_interest_evolution(self):
        num_candidate = 10
        env = RecSim(
            num_candidates=num_candidate, slate_size=3, resample_documents=False
        )
        obs_preprocessor = env.get_obs_preprocessor()
        obs = env.reset()
        state = obs_preprocessor(obs)
        self.assertFalse(state.has_float_features_only)
        self.assertEqual(state.float_features.shape, (1, obs["user"].shape[0]))
        self.assertEqual(state.float_features.dtype, torch.float32)
        self.assertEqual(state.float_features.device, torch.device("cpu"))
        npt.assert_array_almost_equal(obs["user"], state.float_features.squeeze(0))
        doc_float_features = state.candidate_docs.float_features
        self.assertIsNotNone(doc_float_features)
        self.assertEqual(
            doc_float_features.shape, (1, num_candidate, obs["doc"]["0"].shape[0])
        )
        self.assertEqual(doc_float_features.dtype, torch.float32)
        self.assertEqual(doc_float_features.device, torch.device("cpu"))
        for i, v in enumerate(obs["doc"].values()):
            npt.assert_array_almost_equal(v, doc_float_features[0, i])

    @unittest.skipIf(not HAS_RECSIM, "Recsim is not installed")
    def test_recsim_interest_exploration(self):
        num_candidate = 10
        env = RecSim(
            num_candidates=num_candidate,
            slate_size=3,
            resample_documents=False,
            is_interest_exploration=True,
        )
        obs_preprocessor = env.get_obs_preprocessor()
        obs = env.reset()
        state = obs_preprocessor(obs)
        self.assertFalse(state.has_float_features_only)
        self.assertEqual(state.float_features.shape, (1, obs["user"].shape[0]))
        self.assertEqual(state.float_features.dtype, torch.float32)
        self.assertEqual(state.float_features.device, torch.device("cpu"))
        npt.assert_array_almost_equal(obs["user"], state.float_features.squeeze(0))
        doc_float_features = state.candidate_docs.float_features
        self.assertIsNotNone(doc_float_features)

        quality_len = 1
        expected_doc_feature_length = (
            env.observation_space["doc"]["0"]["cluster_id"].n + quality_len
        )

        self.assertEqual(
            doc_float_features.shape, (1, num_candidate, expected_doc_feature_length)
        )
        self.assertEqual(doc_float_features.dtype, torch.float32)
        self.assertEqual(doc_float_features.device, torch.device("cpu"))
        for i, v in enumerate(obs["doc"].values()):
            expected_doc_feature = torch.cat(
                [
                    F.one_hot(torch.tensor(v["cluster_id"]), 2).float(),
                    # This needs unsqueeze because it's a scalar
                    torch.tensor(v["quality"]).unsqueeze(0).float(),
                ],
                dim=0,
            )
            npt.assert_array_almost_equal(
                expected_doc_feature, doc_float_features[0, i]
            )
