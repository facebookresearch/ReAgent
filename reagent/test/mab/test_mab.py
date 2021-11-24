#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from io import BytesIO
from itertools import cycle
from unittest import mock

import numpy as np
import numpy.testing as npt
import torch
from parameterized.parameterized import parameterized
from reagent.mab.mab_algorithm import (
    get_arm_indices,
    place_values_at_indices,
    reindex_multiple_tensors,
    randomized_argmax,
)
from reagent.mab.simulation import (
    BernoilliMAB,
    single_evaluation_bandit_algo,
    multiple_evaluations_bandit_algo,
    compare_bandit_algos,
)
from reagent.mab.thompson_sampling import (
    BaseThompsonSampling,
    NormalGammaThompson,
    BernoulliBetaThompson,
)
from reagent.mab.ucb import (
    BaseUCB,
    MetricUCB,
    UCB1,
    UCBTuned,
)

ALL_UCB_ALGOS = [
    ["MetricUCB", MetricUCB],
    ["UCB1", UCB1],
    ["UCBTuned", UCBTuned],
]

ALL_THOMPSON_ALGOS = [
    ["NormalGammaThompson", NormalGammaThompson],
    ["BernoulliBetaThompson", BernoulliBetaThompson],
]

ALL_MAB_ALGOS = ALL_UCB_ALGOS + ALL_THOMPSON_ALGOS


class TestMAButils(unittest.TestCase):
    def test_get_arm_indices_happy_case(self):
        ids_of_all_arms = ["a", "b", "c", "z", "4"]
        ids_of_arms_in_batch = ["z", "4", "b"]
        idxs = get_arm_indices(ids_of_all_arms, ids_of_arms_in_batch)
        self.assertListEqual(idxs, [3, 4, 1])

    def test_get_arm_indices_fail(self):
        ids_of_all_arms = ["a", "b", "c", "z", "4"]
        ids_of_arms_in_batch = ["z", "4", "b", "o"]
        with self.assertRaises(ValueError):
            get_arm_indices(ids_of_all_arms, ids_of_arms_in_batch)

    def test_place_values_at_indices(self):
        values = torch.tensor([3, 7, 11], dtype=torch.float)
        idxs = [2, 3, 5]
        len_ = 7
        result = place_values_at_indices(values, idxs, len_)
        expected_result = torch.Tensor([0, 0, 3, 7, 0, 11, 0])
        npt.assert_array_equal(result.numpy(), expected_result.numpy())

    def test_reindex_multiple_tensors(self):
        values = (
            torch.tensor([3, 7, 11], dtype=torch.float),
            torch.tensor([4, 2, 89], dtype=torch.float),
        )
        all_ids = ["a", "b", "c", "z", "4"]
        batch_ids = ["z", "4", "b"]
        reindexed_values = reindex_multiple_tensors(all_ids, batch_ids, values)
        npt.assert_equal(
            reindexed_values[0].numpy(), np.array([0.0, 11.0, 0.0, 3.0, 7.0])
        )
        npt.assert_equal(
            reindexed_values[1].numpy(), np.array([0.0, 89.0, 0.0, 4.0, 2.0])
        )

    def _test_randomized_argmax(self, x, expected_idxs):
        best_idxs = set()
        for _ in range(1000):
            best_idxs.add(randomized_argmax(x))
        self.assertSetEqual(best_idxs, expected_idxs)

    def test_randomized_argmax(self):
        self._test_randomized_argmax(torch.tensor([1, 2, 3, 2, 3, 1, 3]), {2, 4, 6})
        self._test_randomized_argmax(
            torch.tensor(
                [1, torch.tensor(float("inf")), 3, 2, 3, torch.tensor(float("inf")), 3]
            ),
            {1, 5},
        )
        self._test_randomized_argmax(
            torch.tensor(
                [
                    torch.tensor(float("inf")),
                    torch.tensor(float("inf")),
                    torch.tensor(float("inf")),
                ]
            ),
            {0, 1, 2},
        )
        self._test_randomized_argmax(torch.tensor([1, 2, 3, 2, 3, 1, 5]), {6})


class TestMAB(unittest.TestCase):
    @parameterized.expand(ALL_MAB_ALGOS)
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

            b.add_batch_observations(
                n_obs_per_arm,
                n_success_per_arm,
                n_success_per_arm,  # squared rewards are same as rewards
            )

            npt.assert_array_equal(
                b.total_n_obs_per_arm.numpy(), total_obs_per_arm.numpy()
            )  # observation counters are correct
            npt.assert_array_equal(
                b.total_sum_reward_per_arm.numpy(), total_success_per_arm.numpy()
            )  # total reward counters are corect
            npt.assert_array_equal(
                b.total_sum_reward_squared_per_arm.numpy(),
                total_success_per_arm.numpy(),
            )  # squared rewards equal to rewards for Bernoulli bandit

            self.assertEqual(
                b.total_n_obs_all_arms, total_obs_per_arm.sum().item()
            )  # total observation counter correct

            avg_rewards = total_success_per_arm / total_obs_per_arm
            npt.assert_allclose(
                b.get_avg_reward_values().numpy(), avg_rewards.numpy()
            )  # avg rewards computed correctly

            scores = b.get_scores()
            forward_scores = b()

            # scores shape and type are correct
            self.assertEqual(scores.shape, (n_arms,))
            self.assertIsInstance(scores, torch.Tensor)
            self.assertEqual(forward_scores.shape, (n_arms,))
            self.assertIsInstance(forward_scores, torch.Tensor)

            if isinstance(b, BaseUCB):
                npt.assert_array_less(
                    avg_rewards,
                    scores.numpy(),
                )  # UCB scores greater than avg rewards

                valid_indices = b.total_n_obs_per_arm.numpy() >= b.min_num_obs_per_arm
                npt.assert_array_equal(
                    scores[valid_indices], forward_scores[valid_indices]
                )

    @parameterized.expand(ALL_MAB_ALGOS)
    def test_class_method(self, name, cls):
        n_arms = 5
        n_obs_per_arm = torch.randint(0, 50, size=(n_arms,)).float()
        n_success_per_arm = torch.rand(size=(n_arms,)) * n_obs_per_arm
        scores = cls.get_scores_from_batch(
            n_obs_per_arm, n_success_per_arm, n_success_per_arm
        )

        # UCB scores shape and type are correct
        self.assertEqual(scores.shape, (n_arms,))
        self.assertIsInstance(scores, torch.Tensor)

        if issubclass(cls, BaseUCB):
            avg_rewards = n_success_per_arm / n_obs_per_arm

            npt.assert_array_less(
                avg_rewards.numpy(),
                np.where(
                    n_obs_per_arm.numpy() >= 1,
                    scores.numpy(),
                    np.nan,
                ),
            )  # UCB scores greater than avg rewards

    @parameterized.expand(ALL_MAB_ALGOS)
    def test_online_training(self, name, cls):
        n_arms = 5
        total_n_obs = 100
        min_num_obs_per_arm = 15
        b = cls(n_arms=n_arms, min_num_obs_per_arm=min_num_obs_per_arm)
        total_obs_per_arm = torch.zeros(n_arms)
        total_success_per_arm = torch.zeros(n_arms)
        true_ctrs = torch.rand(size=(n_arms,))
        for _ in range(total_n_obs):
            chosen_arm = b.get_action()
            reward = torch.bernoulli(true_ctrs[int(chosen_arm)])
            b.add_single_observation(chosen_arm, reward)
            total_obs_per_arm[int(chosen_arm)] += 1
            total_success_per_arm[int(chosen_arm)] += reward
        # each arm has at least the required number of observations
        self.assertLessEqual(min_num_obs_per_arm, b.total_n_obs_per_arm.min().item())
        online_scores = b()
        offline_scores = cls.get_scores_from_batch(
            total_obs_per_arm, total_success_per_arm, total_success_per_arm
        )
        if isinstance(b, BaseUCB):
            npt.assert_array_equal(
                online_scores.numpy(), offline_scores.numpy()
            )  # UCB scores computed by online and offline algorithms match
        elif isinstance(b, NormalGammaThompson):
            b_batch = cls(n_arms=n_arms)
            b_batch.add_batch_observations(
                total_obs_per_arm,
                total_success_per_arm,
                total_success_per_arm,  # squared rewards are same as rewards
            )

            # make sure that posterior parameters are the same
            npt.assert_allclose(
                b_batch.gamma_rates.numpy(), b.gamma_rates.numpy(), rtol=1e-5
            )
            npt.assert_allclose(b_batch.mus.numpy(), b.mus.numpy(), rtol=1e-5)
            npt.assert_array_equal(
                b_batch.total_n_obs_per_arm.numpy(), b.total_n_obs_per_arm.numpy()
            )
            npt.assert_array_equal(
                b_batch.total_sum_reward_per_arm.numpy(),
                b.total_sum_reward_per_arm.numpy(),
            )
            npt.assert_array_equal(
                b_batch.total_sum_reward_squared_per_arm.numpy(),
                b.total_sum_reward_squared_per_arm.numpy(),
            )

        elif isinstance(b, BaseThompsonSampling):
            npt.assert_raises(
                AssertionError,
                npt.assert_array_equal,
                online_scores.numpy(),
                offline_scores.numpy(),
            )
            # Thompson sampling scores are stochastic, so shouldn't be equal

    @parameterized.expand(ALL_MAB_ALGOS)
    def test_save_load(self, name, cls):
        n_arms = 5
        b = cls(n_arms=n_arms)
        n_obs_per_arm = torch.randint(0, 100, size=(n_arms,)).float()
        n_success_per_arm = torch.rand(size=(n_arms,)) * n_obs_per_arm
        b.add_batch_observations(n_obs_per_arm, n_success_per_arm, n_success_per_arm)

        avg_rewards_before_save = b.get_avg_reward_values()

        if isinstance(b, BaseUCB):
            ucb_scores_before_save = b.get_scores()

        f_write = BytesIO()
        torch.save(b, f_write)
        f_write.seek(0)
        f_read = BytesIO(f_write.read())
        f_write.close()
        b_loaded = torch.load(f_read)
        f_read.close()

        if isinstance(b, BaseUCB):
            ucb_scores_after_load = b_loaded.get_scores()
            npt.assert_array_equal(
                ucb_scores_before_save.numpy(), ucb_scores_after_load.numpy()
            )  # UCB scores are same before saving and after loading

        avg_rewards_after_load = b_loaded.get_avg_reward_values()
        npt.assert_array_equal(
            avg_rewards_before_save.numpy(), avg_rewards_after_load.numpy()
        )  # avg rewards are same before saving and after loading

        self.assertListEqual(b.arm_ids, b_loaded.arm_ids)

    @parameterized.expand(ALL_MAB_ALGOS)
    def test_custom_arm_ids(self, name, cls):
        # arm 0 earns no rewards, so we specify arm_ids 1,...,N explicitly
        n_arms = 5
        b = cls(n_arms=n_arms)
        n_obs_per_arm = torch.randint(0, 100, size=(n_arms - 1,)).float()
        n_success_per_arm = torch.rand(size=(n_arms - 1,)) * n_obs_per_arm
        b.add_batch_observations(
            n_obs_per_arm,
            n_success_per_arm,
            n_success_per_arm,
            arm_ids=list(map(str, range(1, n_arms))),
        )

        self.assertEqual(b.total_n_obs_per_arm[0], 0)
        npt.assert_array_equal(n_obs_per_arm.numpy(), b.total_n_obs_per_arm[1:].numpy())
        npt.assert_array_equal(
            n_success_per_arm.numpy(), b.total_sum_reward_per_arm[1:].numpy()
        )
        npt.assert_array_equal(
            n_success_per_arm.numpy(),
            b.total_sum_reward_squared_per_arm[1:].numpy(),
        )


class TestSimulation(unittest.TestCase):
    def test_single_evaluation(self):
        bandit = BernoilliMAB(100, torch.tensor([0.3, 0.5]))
        algo = UCB1(n_arms=2)
        regret_trajectory = single_evaluation_bandit_algo(bandit, algo)

        self.assertIsInstance(regret_trajectory, np.ndarray)
        self.assertEqual(regret_trajectory.shape, (bandit.max_steps,))

        # make sure regret is non-decreasing
        self.assertGreaterEqual(np.diff(regret_trajectory, prepend=0).min(), 0)

    def test_single_evaluation_update_every(self):
        num_steps = 100
        update_every = 10

        bandit = BernoilliMAB(num_steps, torch.tensor([0.3, 0.5]))
        algo = UCB1(n_arms=2)
        algo.add_batch_observations = mock.Mock()
        algo.get_action = mock.Mock(side_effect=cycle(["0", "1"]))
        regret_trajectory = single_evaluation_bandit_algo(
            bandit, algo, update_every=update_every, freeze_scores_btw_updates=False
        )
        self.assertEqual(len(regret_trajectory), num_steps)
        self.assertEqual(
            algo.add_batch_observations.call_count, num_steps / update_every
        )
        self.assertEqual(algo.get_action.call_count, num_steps)

        bandit = BernoilliMAB(num_steps, torch.tensor([0.3, 0.5]))
        algo = UCB1(n_arms=2)
        algo.add_batch_observations = mock.Mock()
        algo.get_action = mock.Mock(side_effect=cycle(["0", "1"]))
        regret_trajectory = single_evaluation_bandit_algo(
            bandit, algo, update_every=update_every, freeze_scores_btw_updates=True
        )
        self.assertEqual(len(regret_trajectory), num_steps)
        self.assertEqual(
            algo.add_batch_observations.call_count, num_steps / update_every
        )
        self.assertEqual(algo.get_action.call_count, num_steps / update_every)

    def test_multiple_evaluations_bandit_algo(self):
        max_steps = 20
        regret_trajectory = multiple_evaluations_bandit_algo(
            algo_cls=UCB1,
            bandit_cls=BernoilliMAB,
            n_bandits=3,
            max_steps=max_steps,
            algo_kwargs={"n_arms": 2},
            bandit_kwargs={"probs": torch.Tensor([0.3, 0.5])},
        )

        self.assertIsInstance(regret_trajectory, np.ndarray)
        self.assertEqual(regret_trajectory.shape, (max_steps,))

        # make sure regret is non-decreasing
        self.assertGreaterEqual(np.diff(regret_trajectory, prepend=0).min(), 0)

    def test_compare_bandit_algos(self):
        max_steps = 1000
        algo_clss = [UCB1, MetricUCB, BernoulliBetaThompson]
        algo_names, regret_trajectories = compare_bandit_algos(
            algo_clss=algo_clss,
            bandit_cls=BernoilliMAB,
            n_bandits=5,
            max_steps=max_steps,
            algo_kwargs={"n_arms": 2},
            bandit_kwargs={"probs": torch.Tensor([0.1, 0.2])},
        )

        self.assertEqual(len(algo_names), len(algo_clss))
        self.assertEqual(len(regret_trajectories), len(algo_clss))

        self.assertListEqual(algo_names, ["UCB1", "MetricUCB", "BernoulliBetaThompson"])

        for traj in regret_trajectories:
            self.assertIsInstance(traj, np.ndarray)
            self.assertEqual(traj.shape, (max_steps,))

            # make sure regret is non-decreasing
            self.assertGreaterEqual(np.diff(traj, prepend=0).min(), 0)

        # UCB1 should be much worse than MetricUCB in this setting
        self.assertGreater(regret_trajectories[0][-1], regret_trajectories[1][-1])
