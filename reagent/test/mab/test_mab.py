import unittest
from io import BytesIO

import numpy as np
import numpy.testing as npt
import torch
from parameterized import parameterized
from reagent.mab.mab_algorithm import (
    get_arm_indices,
    place_values_at_indices,
)
from reagent.mab.thompson_sampling import (
    BaseThompsonSampling,
    NormalGammaThompson,
    BernoulliBetaThompson,
)
from reagent.mab.ucb import (
    BaseUCB,
    MetricUCB,
    UCBTuned,
    UCB1,
)

ALL_UCB_ALGOS = [
    ["MetricUCB", MetricUCB],
    ["UCBTuned", UCBTuned],
    ["UCB1", UCB1],
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

            if isinstance(b, BaseUCB):
                ucb_scores = b.get_ucb_scores()
                forward_scores = b()

                # UCB scores shape and type are correct
                self.assertEqual(ucb_scores.shape, (n_arms,))
                self.assertIsInstance(ucb_scores, torch.Tensor)

                npt.assert_array_less(
                    avg_rewards,
                    np.where(
                        b.total_n_obs_per_arm.numpy() > 0, ucb_scores.numpy(), np.nan
                    ),
                )  # UCB scores greater than avg rewards

                npt.assert_array_equal(ucb_scores, forward_scores)

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
                np.where(n_obs_per_arm.numpy() > 0, scores.numpy(), np.nan),
            )  # UCB scores greater than avg rewards

    @parameterized.expand(ALL_MAB_ALGOS)
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
            ucb_scores_before_save = b.get_ucb_scores()

        f_write = BytesIO()
        torch.save(b, f_write)
        f_write.seek(0)
        f_read = BytesIO(f_write.read())
        f_write.close()
        b_loaded = torch.load(f_read)
        f_read.close()

        if isinstance(b, BaseUCB):
            ucb_scores_after_load = b_loaded.get_ucb_scores()
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
            arm_ids=list(range(1, n_arms)),
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
