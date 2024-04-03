#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.utils import add_importance_weights


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
            action_log_probability=torch.log(
                torch.tensor([[0.5], [0.5]], dtype=torch.float)
            ),
        )

    def test_add_importance_weights(self):
        model_actions = torch.tensor([[1], [1]], dtype=torch.long)
        new_batch = add_importance_weights(self.batch, model_actions)
        # everything except weights should remain the same in the new batch
        for name in ["context_arm_features", "action", "reward"]:
            npt.assert_allclose(
                getattr(self.batch, name).numpy(), getattr(new_batch, name).numpy()
            )

        # data weights should be unchanged
        npt.assert_allclose(new_batch.weight.numpy(), self.batch.weight.numpy())
        # weights should be zero-ed out where action!= model_action
        self.assertEqual(new_batch.effective_weight[0, 0].item(), 0.0)
        # weights should be multiplied inverse probability of logged action where action == model_action
        self.assertEqual(
            new_batch.effective_weight[1, 0].item(),
            self.batch.weight[1, 0].item() * 2,
        )
