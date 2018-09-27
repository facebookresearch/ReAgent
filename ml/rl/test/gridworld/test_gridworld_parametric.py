#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_continuous_enum import GridworldContinuousEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    FactorizationParameters,
    FeedForwardParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


class TestGridworldParametric(unittest.TestCase):
    def setUp(self):
        self.minibatch_size = 512
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

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
                learning_rate=0.05,
                optimizer="ADAM",
            ),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
            in_training_cpe=InTrainingCPEParameters(mdp_sampled_rate=0.1),
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
                # These are used by reward network
                layers=[-1, 256, 128, -1],
                activations=["relu", "relu", "linear"],
                factorization_parameters=FactorizationParameters(
                    state=FeedForwardParameters(
                        layers=[-1, 128, 64, 32], activations=["relu", "relu", "linear"]
                    ),
                    action=FeedForwardParameters(
                        layers=[-1, 128, 64, 32], activations=["relu", "relu", "linear"]
                    ),
                ),
                minibatch_size=self.minibatch_size,
                learning_rate=0.05,
                optimizer="ADAM",
            ),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
            in_training_cpe=InTrainingCPEParameters(mdp_sampled_rate=0.1),
        )

    def get_sarsa_trainer(self, environment, parameters=None):
        parameters = parameters or self.get_sarsa_parameters()
        return ParametricDQNTrainer(
            parameters, environment.normalization, environment.normalization_action
        )

    def test_trainer_sarsa(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_factorized(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum(self):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum_factorized(self):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_evaluator_ground_truth(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(None, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)

    def test_evaluator_ground_truth_factorized(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = true_values
        trainer = self.get_sarsa_trainer(
            environment, self.get_sarsa_parameters_factorized()
        )
        evaluator = Evaluator(None, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)
