import unittest
from io import BytesIO

import numpy as np
import numpy.testing as npt
import torch
from parameterized import parameterized
from reagent.mab.ucb import (
    UCBTunedBernoulli,
    MetricUCB,
    UCBTuned,
    UCB1,
    _get_arm_indices,
    _place_values_at_indices,
)


class TestUCButils(unittest.TestCase):
    def test_get_arm_indices_happy_case(self):
        ids_of_all_arms = ["a", "b", "c", "z", "4"]
        ids_of_arms_in_batch = ["z", "4", "b"]
        idxs = _get_arm_indices(ids_of_all_arms, ids_of_arms_in_batch)
        self.assertListEqual(idxs, [3, 4, 1])

    def test_get_arm_indices_fail(self):
        ids_of_all_arms = ["a", "b", "c", "z", "4"]
        ids_of_arms_in_batch = ["z", "4", "b", "o"]
        with self.assertRaises(ValueError):
            _get_arm_indices(ids_of_all_arms, ids_of_arms_in_batch)

    def test_place_values_at_indices(self):
        values = torch.tensor([3, 7, 11], dtype=torch.float)
        idxs = [2, 3, 5]
        len_ = 7
        result = _place_values_at_indices(values, idxs, len_)
        expected_result = torch.Tensor([0, 0, 3, 7, 0, 11, 0])
        npt.assert_array_equal(result.numpy(), expected_result.numpy())


class TestUCB(unittest.TestCase):
    @parameterized.expand(
        [
            ["UCBTunedBernoulli", UCBTunedBernoulli],
            ["MetricUCB", MetricUCB],
            ["UCBTuned", UCBTuned],
            ["UCB1", UCB1],
        ]
    )
    def test_batch_training(self, name, cls):
        n_arms = 5
        b = cls(n_arms=n_arms)
        total_obs_per_arm = torch.zeros(n_arms)
        total_success_per_arm = torch.zeros(n_arms)
        for _ in range(10):
            n_obs_per_arm = torch.randint(0, 50, size=(n_arms,)).float()
            n_success_per_arm = torch.rand(size=(n_arms,)) * n_obs_per_arm
            total_obs_per_arm += n_obs_per_arm
            total_success_per_arm += n_success_per_arm

            if cls == UCBTuned:
                # UCBTuned retquires additional input
                b.add_batch_observations(
                    n_obs_per_arm, n_success_per_arm, n_success_per_arm
                )
            else:
                b.add_batch_observations(n_obs_per_arm, n_success_per_arm)

            npt.assert_array_equal(
                b.total_n_obs_per_arm.numpy(), total_obs_per_arm.numpy()
            )  # observation counters are correct
            npt.assert_array_equal(
                b.total_sum_reward_per_arm.numpy(), total_success_per_arm.numpy()
            )  # success counters are corect
            if issubclass(cls, UCBTuned):
                # we keep track of squared rewards only for UCBTuned
                npt.assert_array_equal(
                    b.total_sum_reward_squared_per_arm.numpy(),
                    total_success_per_arm.numpy(),
                )  # squared rewards equal to rewards for Bernoulli bandit

            self.assertEqual(
                b.total_n_obs_all_arms, total_obs_per_arm.sum().item()
            )  # total observation counter correct

            ucb_scores = b.get_ucb_scores()

            # UCB scores shape and type are correct
            self.assertEqual(ucb_scores.shape, (n_arms,))
            self.assertIsInstance(ucb_scores, torch.Tensor)

            avg_rewards = total_success_per_arm / total_obs_per_arm

            npt.assert_array_almost_equal(
                b.get_avg_reward_values().numpy(), avg_rewards.numpy()
            )  # avg rewards computed correctly

            npt.assert_array_less(
                avg_rewards,
                np.where(b.total_n_obs_per_arm.numpy() > 0, ucb_scores.numpy(), np.nan),
            )  # UCB scores greater than avg rewards

    @parameterized.expand(
        [
            ["UCBTunedBernoulli", UCBTunedBernoulli],
            ["MetricUCB", MetricUCB],
            ["UCBTuned", UCBTuned],
            ["UCB1", UCB1],
        ]
    )
    def test_class_method(self, name, cls):
        n_arms = 5
        n_obs_per_arm = torch.randint(0, 50, size=(n_arms,)).float()
        n_success_per_arm = torch.rand(size=(n_arms,)) * n_obs_per_arm
        if cls == UCBTuned:
            ucb_scores = cls.get_ucb_scores_from_batch(
                n_obs_per_arm, n_success_per_arm, n_success_per_arm
            )
        else:
            ucb_scores = cls.get_ucb_scores_from_batch(n_obs_per_arm, n_success_per_arm)

        # UCB scores shape and type are correct
        self.assertEqual(ucb_scores.shape, (n_arms,))
        self.assertIsInstance(ucb_scores, torch.Tensor)

        avg_rewards = n_success_per_arm / n_obs_per_arm

        npt.assert_array_less(
            avg_rewards.numpy(),
            np.where(n_obs_per_arm.numpy() > 0, ucb_scores.numpy(), np.nan),
        )  # UCB scores greater than avg rewards

    @parameterized.expand(
        [
            ["UCBTunedBernoulli", UCBTunedBernoulli],
            ["MetricUCB", MetricUCB],
            ["UCBTuned", UCBTuned],
            ["UCB1", UCB1],
        ]
    )
    def test_online_training(self, name, cls):
        n_arms = 5
        total_n_obs = 100
        b = cls(n_arms=n_arms)
        total_obs_per_arm = torch.zeros(n_arms)
        total_success_per_arm = torch.zeros(n_arms)
        true_ctrs = torch.rand(size=(n_arms,))
        for _ in range(total_n_obs):
            chosen_arm = b.get_action()
            reward = torch.bernoulli(true_ctrs[chosen_arm])
            b.add_single_observation(chosen_arm, reward)
            total_obs_per_arm[chosen_arm] += 1
            total_success_per_arm[chosen_arm] += reward

        online_ucb_scores = b.get_ucb_scores()

        if cls == UCBTuned:
            offline_ucb_scores = cls.get_ucb_scores_from_batch(
                total_obs_per_arm, total_success_per_arm, total_success_per_arm
            )
        else:
            offline_ucb_scores = cls.get_ucb_scores_from_batch(
                total_obs_per_arm, total_success_per_arm
            )

        npt.assert_array_equal(
            online_ucb_scores.numpy(), offline_ucb_scores.numpy()
        )  # UCB scores computed by online and offline algorithms match

    @parameterized.expand(
        [
            ["UCBTunedBernoulli", UCBTunedBernoulli],
            ["MetricUCB", MetricUCB],
            ["UCBTuned", UCBTuned],
            ["UCB1", UCB1],
        ]
    )
    def test_save_load(self, name, cls):
        n_arms = 5
        b = cls(n_arms=n_arms)
        n_obs_per_arm = torch.randint(0, 100, size=(n_arms,)).float()
        n_success_per_arm = torch.rand(size=(n_arms,)) * n_obs_per_arm
        if cls == UCBTuned:
            # UCBTuned retquires additional input
            b.add_batch_observations(
                n_obs_per_arm, n_success_per_arm, n_success_per_arm
            )
        else:
            b.add_batch_observations(n_obs_per_arm, n_success_per_arm)

        ucb_scores_before_save = b.get_ucb_scores()

        f_write = BytesIO()
        torch.save(b, f_write)
        f_write.seek(0)
        f_read = BytesIO(f_write.read())
        f_write.close()
        b_loaded = torch.load(f_read)
        f_read.close()

        ucb_scores_after_load = b_loaded.get_ucb_scores()

        npt.assert_array_equal(
            ucb_scores_before_save.numpy(), ucb_scores_after_load.numpy()
        )  # UCB scores are same before saving and after loading

        self.assertListEqual(b.arm_ids, b_loaded.arm_ids)

    @parameterized.expand(
        [
            ["UCBTunedBernoulli", UCBTunedBernoulli],
            ["MetricUCB", MetricUCB],
            ["UCBTuned", UCBTuned],
            ["UCB1", UCB1],
        ]
    )
    def test_custom_arm_ids(self, name, cls):
        # arm 0 earns no rewards, so we specify arm_ids 1,...,N explicitly
        n_arms = 5
        b = cls(n_arms=n_arms)
        n_obs_per_arm = torch.randint(0, 100, size=(n_arms - 1,)).float()
        n_success_per_arm = torch.rand(size=(n_arms - 1,)) * n_obs_per_arm
        if cls == UCBTuned:
            # UCBTuned requires additional input
            b.add_batch_observations(
                n_obs_per_arm,
                n_success_per_arm,
                n_success_per_arm,
                arm_ids=list(range(1, n_arms)),
            )
        else:
            b.add_batch_observations(
                n_obs_per_arm, n_success_per_arm, arm_ids=list(range(1, n_arms))
            )

        self.assertEqual(b.total_n_obs_per_arm[0], 0)
        npt.assert_array_equal(n_obs_per_arm.numpy(), b.total_n_obs_per_arm[1:].numpy())
        npt.assert_array_equal(
            n_success_per_arm.numpy(), b.total_sum_reward_per_arm[1:].numpy()
        )
        if issubclass(cls, UCBTuned):
            npt.assert_array_equal(
                n_success_per_arm.numpy(),
                b.total_sum_reward_squared_per_arm[1:].numpy(),
            )
