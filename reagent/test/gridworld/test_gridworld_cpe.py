#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random

import numpy as np
import torch
from ml.rl.evaluation.doubly_robust_estimator import DoublyRobustEstimator
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.evaluation.sequential_doubly_robust_estimator import (
    SequentialDoublyRobustEstimator,
)
from ml.rl.evaluation.weighted_sequential_doubly_robust_estimator import (
    WeightedSequentialDoublyRobustEstimator,
)
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase


logger = logging.getLogger(__name__)


class TestGridworldCPE(GridworldTestBase):
    NUM_J_STEPS_FOR_MAGIC_ESTIMATOR = 25

    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def test_doubly_robust(self):
        """Both the logged and model policies are epsilon-greedy policies where
        greedy = optimal, but the epsilon values are different. We test a variety
        of epsilon pairs to check the estimator's ability to evaluate model policies
        that are much different than the logged policies that generated the data. By
        computing the true values associated with both epsilon policies, we can
        see the performance and compute a percentage error.
        """
        environment = Gridworld()
        dr = DoublyRobustEstimator()
        epsilon_test_pairs = [
            [1.0, 0.05],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.05, 1.0],
        ]
        for epsilon_pair in epsilon_test_pairs:
            epsilon_logged = epsilon_pair[0]
            epsilon_model = epsilon_pair[1]
            samples_logged = environment.generate_samples(
                10000, epsilon_logged, DISCOUNT
            )
            edp = self.create_edp(environment, samples_logged, epsilon_model)
            cpe_drs = dr.estimate(edp)
            true_logged_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_logged
            )
            true_model_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_model
            )
            ratio = true_model_value[0] / true_logged_value[0]
            cpe_drs_names = [
                "One-step direct method",
                "One-step inverse propensity",
                "One-step doubly robust",
            ]
            for i in range(len(cpe_drs)):
                percent_err = (cpe_drs[i].normalized - ratio) / ratio * 100
                logger.info(
                    cpe_drs_names[i]
                    + ": epsilon_pair = ("
                    + str(epsilon_logged)
                    + ", "
                    + str(epsilon_model)
                    + ");\n"
                    + "true ratio = "
                    + str(ratio)
                    + ", computed ratio = "
                    + str(cpe_drs[i].normalized)
                    + ", percent error = "
                    + str(percent_err)
                    + "."
                )
                self.assertLessEqual(np.absolute(percent_err), 1000)
                self.assertLessEqual(
                    cpe_drs[i].normalized_std_error, cpe_drs[i].normalized
                )

    def test_sequential_doubly_robust(self):
        """Both the logged and model policies are epsilon-greedy policies where
        greedy = optimal, but the epsilon values are different. We test a variety
        of epsilon pairs to check the estimator's ability to evaluate model policies
        that are much different than the logged policies that generated the data. By
        computing the true values associated with both epsilon policies, we can
        see the performance and compute a percentage error.
        """
        environment = Gridworld()
        sequential_dr = SequentialDoublyRobustEstimator(DISCOUNT)
        epsilon_test_pairs = [
            [1.0, 0.05],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.05, 1.0],
        ]
        for epsilon_pair in epsilon_test_pairs:
            epsilon_logged = epsilon_pair[0]
            epsilon_model = epsilon_pair[1]
            samples_logged = environment.generate_samples(
                10000, epsilon_logged, DISCOUNT
            )
            edp = self.create_edp(environment, samples_logged, epsilon_model)
            cpe_sequential_dr = sequential_dr.estimate(edp)
            true_logged_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_logged
            )
            true_model_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_model
            )
            ratio = true_model_value[0] / true_logged_value[0]
            percent_err = (cpe_sequential_dr.normalized - ratio) / ratio * 100
            logger.info(
                "Sequential DR: epsilon_pair = ("
                + str(epsilon_logged)
                + ", "
                + str(epsilon_model)
                + ");\n"
                + "true ratio = "
                + str(ratio)
                + ", computed ratio = "
                + str(cpe_sequential_dr.normalized)
                + ", percent error = "
                + str(percent_err)
                + "."
            )
            self.assertLessEqual(np.absolute(percent_err), 100)
            self.assertLessEqual(
                cpe_sequential_dr.normalized_std_error, cpe_sequential_dr.normalized
            )

    def test_magic(self):
        """Both the logged and model policies are epsilon-greedy policies where
        greedy = optimal, but the epsilon values are different. We test a variety
        of epsilon pairs to check the estimator's ability to evaluate model policies
        that are much different than the logged policies that generated the data. By
        computing the true values associated with both epsilon policies, we can
        see the performance and compute a percentage error.
        """
        environment = Gridworld()
        weighted_sequential_dr = WeightedSequentialDoublyRobustEstimator(DISCOUNT)
        epsilon_test_pairs = [
            [1.0, 0.05],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.05, 1.0],
        ]
        for epsilon_pair in epsilon_test_pairs:
            epsilon_logged = epsilon_pair[0]
            epsilon_model = epsilon_pair[1]
            samples_logged = environment.generate_samples(
                10000, epsilon_logged, DISCOUNT
            )
            edp = self.create_edp(environment, samples_logged, epsilon_model)
            cpe_magic = weighted_sequential_dr.estimate(
                edp, TestGridworldCPE.NUM_J_STEPS_FOR_MAGIC_ESTIMATOR, True
            )
            true_logged_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_logged
            )
            true_model_value = environment.true_q_epsilon_values(
                DISCOUNT, epsilon_model
            )
            ratio = true_model_value[0] / true_logged_value[0]
            percent_err = (cpe_magic.normalized - ratio) / ratio * 100
            logger.info(
                "Magic: epsilon_pair = ("
                + str(epsilon_logged)
                + ", "
                + str(epsilon_model)
                + ");\n"
                + "true ratio = "
                + str(ratio)
                + ", computed ratio = "
                + str(cpe_magic.normalized)
                + ", percent error = "
                + str(percent_err)
                + "."
            )
            self.assertLessEqual(np.absolute(percent_err), 100)
            self.assertLessEqual(cpe_magic.normalized_std_error, cpe_magic.normalized)

    def create_edp(self, environment, samples, epsilon_model):
        """Generate a EvaluationDataPage such that the model policy is epsilon
        greedy with parameter epsilon_model. The true values of this policy are
        used for the model_values* data.
        """
        tdp = environment.preprocess_samples(
            samples, len(samples.mdp_ids), do_shuffle=False
        )[0]
        # compute rewards, probs, values for all actions of each sampled state
        model_rewards = environment.true_rewards_all_actions_for_sample(samples.states)
        model_propensities = environment.policy_probabilities_for_sample(
            samples.states, epsilon_model
        )
        model_values = environment.true_epsilon_values_all_actions_for_sample(
            samples.states, epsilon_model
        )
        # compute rewards for logged action
        model_rewards_logged_action = environment.true_rewards_for_sample(
            samples.states, samples.actions
        )
        edp = EvaluationDataPage(
            mdp_id=np.array(samples.mdp_ids).reshape(-1, 1),
            sequence_number=torch.tensor(samples.sequence_numbers, dtype=torch.int),
            logged_propensities=tdp.propensities,
            logged_rewards=tdp.rewards,
            action_mask=tdp.actions,
            model_propensities=torch.tensor(model_propensities, dtype=torch.float32),
            model_rewards=torch.tensor(model_rewards, dtype=torch.float32),
            model_rewards_for_logged_action=torch.tensor(
                model_rewards_logged_action, dtype=torch.float32
            ),
            model_values=torch.tensor(model_values, dtype=torch.float32),
            model_values_for_logged_action=None,
            possible_actions_mask=tdp.possible_actions_mask,
        )
        return edp
