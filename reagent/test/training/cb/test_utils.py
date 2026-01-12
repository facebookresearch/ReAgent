# pyre-unsafe
import unittest

import numpy as np
import torch
from numpy import testing as npt
from reagent.core.types import CBInput
from reagent.training.cb.utils import (
    add_chosen_arm_features,
    argmax_random_tie_breaks,
    get_model_actions,
)


class TestCButils(unittest.TestCase):
    def test_add_chosen_arm_features(self):
        all_arms_features = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float
        )
        arm_ids = torch.tensor([[10, 13], [15, 20]], dtype=torch.long)
        actions = torch.tensor([[1], [0]], dtype=torch.long)
        batch = CBInput(
            context_arm_features=all_arms_features,
            action=actions,
            arms=arm_ids,
        )
        new_batch = add_chosen_arm_features(batch)
        npt.assert_equal(
            new_batch.features_of_chosen_arm.numpy(), np.array([[3.0, 4.0], [5.0, 6.0]])
        )
        npt.assert_equal(new_batch.chosen_arm_id.numpy(), np.array([[13], [15]]))

    def test_argmax_random_tie_breaks_no_mask(self):
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(argmax_values_returned[0], {1, 2})
        self.assertSetEqual(argmax_values_returned[1], {0, 1})
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(argmax_values_returned[3], {1, 2})

    def test_argmax_random_tie_breaks_mask(self):
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores, mask)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_randomize(self):
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_actions(scores, mask, randomize_ties=True)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_not_randomize(self):
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_actions(scores, mask, randomize_ties=False)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[2],
            {
                0,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )
