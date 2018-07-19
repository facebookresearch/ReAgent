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
        self.minibatch_size = 512
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

    def get_sarsa_parameters(self):
        return ContinuousActionModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=1.0,
                reward_burnin=100,
                maxq_learning=False,
            ),
            training=TrainingParameters(
                layers=[-1, 256, 128, -1],
                activations=["relu", "relu", "linear"],
                minibatch_size=self.minibatch_size,
                learning_rate=0.1,
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
        samples = environment.generate_samples(100000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train_numpy(tdp, None)
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum(self):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(100000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train_numpy(tdp, None)
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_evaluator_ground_truth(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(200000, 1.0)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.reward_timelines = []
        for tv in true_values:
            samples.reward_timelines.append({0: tv})
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(None, 10, DISCOUNT)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train_numpy(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.1)
