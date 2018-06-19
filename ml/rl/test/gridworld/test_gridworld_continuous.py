#!/usr/bin/env python3

import random
import numpy as np
import unittest

from ml.rl.training.evaluator import Evaluator
from ml.rl.thrift.core.ttypes import (
    RLParameters,
    TrainingParameters,
    ContinuousActionModelParameters,
    KnnParameters,
)
from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_continuous_enum import GridworldContinuousEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator


class TestGridworldContinuous(unittest.TestCase):
    def setUp(self):
        self.minibatch_size = 1024
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

    def get_sarsa_parameters(self):
        return ContinuousActionModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=False,
            ),
            training=TrainingParameters(
                layers=[-1, 200, -1],
                activations=["linear", "linear"],
                minibatch_size=self.minibatch_size,
                learning_rate=0.01,
                optimizer="ADAM",
            ),
            knn=KnnParameters(model_type="DQN"),
        )

    def get_sarsa_trainer(self, environment):
        return ContinuousActionDQNTrainer(
            self.get_sarsa_parameters(),
            environment.normalization,
            environment.normalization_action,
        )

    def test_trainer_sarsa(self):
        environment = GridworldContinuous()
        states, actions, propensities, rewards, next_states, next_actions, is_terminal, possible_next_actions, reward_timelines = environment.generate_samples(
            100000, 1.0
        )
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(environment, False)
        tdps = environment.preprocess_samples(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            reward_timelines,
            self.minibatch_size,
        )

        self.assertGreater(evaluator.evaluate(predictor), 0.15)

        for tdp in tdps:
            trainer.train_numpy(tdp, None)
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.05)

    def test_trainer_sarsa_enum(self):
        environment = GridworldContinuousEnum()
        states, actions, propensities, rewards, next_states, next_actions, is_terminal, possible_next_actions, reward_timelines = environment.generate_samples(
            100000, 1.0
        )
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(environment, False)
        tdps = environment.preprocess_samples(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            reward_timelines,
            self.minibatch_size,
        )

        self.assertGreater(evaluator.evaluate(predictor), 0.15)

        for tdp in tdps:
            trainer.train_numpy(tdp, None)
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.05)

    def test_trainer_maxq(self):
        environment = GridworldContinuous()
        rl_parameters = self.get_sarsa_parameters()
        new_rl_parameters = ContinuousActionModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=True,
            ),
            training=rl_parameters.training,
            knn=rl_parameters.knn,
        )
        maxq_trainer = ContinuousActionDQNTrainer(
            new_rl_parameters,
            environment.normalization,
            environment.normalization_action,
        )

        states, actions, propensities, rewards, next_states, next_actions, is_terminal, possible_next_actions, reward_timelines = environment.generate_samples(
            100000, 1.0
        )
        predictor = maxq_trainer.predictor()
        tdps = environment.preprocess_samples(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            reward_timelines,
            self.minibatch_size,
        )
        evaluator = GridworldContinuousEvaluator(environment, True)
        self.assertGreater(evaluator.evaluate(predictor), 0.2)

        for _ in range(2):
            for tdp in tdps:
                maxq_trainer.train_numpy(tdp, None)
            evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_evaluator_ground_truth(self):
        environment = GridworldContinuous()
        states, actions, propensities, rewards, next_states, next_actions, is_terminal, possible_next_actions, _ = environment.generate_samples(
            100000, 1.0
        )
        true_values = environment.true_values_for_sample(states, actions, False)
        # Hijack the reward timeline to insert the ground truth
        reward_timelines = []
        for tv in true_values:
            reward_timelines.append({0: tv})
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator()
        tdps = environment.preprocess_samples(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            reward_timelines,
            self.minibatch_size,
        )

        for tdp in tdps:
            trainer.train_numpy(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.05)
        self.assertLess(evaluator.mc_loss[-1], 0.12)

    def test_evaluator_timeline(self):
        environment = GridworldContinuous()
        states, actions, propensities, rewards, next_states, next_actions, is_terminal, possible_next_actions, reward_timelines = environment.generate_samples(
            100000, 1.0
        )
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator()

        tdps = environment.preprocess_samples(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            reward_timelines,
            self.minibatch_size,
        )
        for tdp in tdps:
            trainer.train_numpy(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.2)
        self.assertLess(evaluator.mc_loss[-1], 0.2)
