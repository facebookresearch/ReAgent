#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.utils import zero_out_skipped_obs_weights


class TestCBEvalUtils(unittest.TestCase):
    def setUp(self):
        self.batch = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2],
                        [1, 3],
                    ],
                    [
                        [1, 4],
                        [1, 5],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
            weight=torch.tensor([[7], [5]], dtype=torch.float),
        )

    def test_zero_out_skipped_obs_weights(self):
        model_actions = torch.tensor([[1], [1]], dtype=torch.long)
        new_batch = zero_out_skipped_obs_weights(self.batch, model_actions)
        # everything except weights should remain the same in the new batch
        for name in ["context_arm_features", "action", "reward"]:
            npt.assert_allclose(
                getattr(self.batch, name).numpy(), getattr(new_batch, name).numpy()
            )

        # weights should be zero-ed out where action!= model_action
        self.assertEqual(new_batch.weight[0, 0].item(), 0.0)
        self.assertEqual(new_batch.weight[1, 0].item(), self.batch.weight[1, 0].item())
