#!/usr/bin/env python3

import random
import unittest

import numpy as np
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_continuous_enum import GridworldContinuousEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    FactorizationParameters,
    FeedForwardParameters,
    KnnParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


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
                learning_rate=0.125,
                optimizer="ADAM",
            ),
            knn=KnnParameters(model_type="DQN"),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
        )

    def get_sarsa_parameters_factorized(self):
        return ContinuousActionModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=1.0,
                reward_burnin=100,
                maxq_learning=False,
            ),
            training=TrainingParameters(
                layers=[],
                activations=[],
                factorization_parameters=FactorizationParameters(
                    state=FeedForwardParameters(
                        layers=[-1, 128, 64, 32], activations=["relu", "relu", "linear"]
                    ),
                    action=FeedForwardParameters(
                        layers=[-1, 128, 64, 32], activations=["relu", "relu", "linear"]
                    ),
                ),
                minibatch_size=self.minibatch_size,
                learning_rate=0.125,
                optimizer="ADAM",
            ),
            knn=KnnParameters(model_type="DQN"),
        )

    def get_sarsa_trainer(self, environment, parameters=None):
        parameters = parameters or self.get_sarsa_parameters()
        return ParametricDQNTrainer(
            parameters, environment.normalization, environment.normalization_action
        )

    def test_trainer_sarsa(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(150000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_factorized(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(150000, 1.0)
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum(self):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(150000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum_factorized(self):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(150000, 1.0)
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp)

        predictor = trainer.predictor()
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
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)

    def test_evaluator_ground_truth_factorized(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(200000, 1.0)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.reward_timelines = []
        for tv in true_values:
            samples.reward_timelines.append({0: tv})
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        evaluator = Evaluator(None, 10, DISCOUNT)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)
