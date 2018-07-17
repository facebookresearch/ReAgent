#!/usr/bin/env python3

import random
import numpy as np
import unittest

from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.evaluator import Evaluator
from ml.rl.thrift.core.ttypes import (
    RLParameters,
    TrainingParameters,
    DiscreteActionModelParameters,
)
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_enum import GridworldEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.test.gridworld.gridworld_base import DISCOUNT


class TestGridworld(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        self.minibatch_size = 1024
        super(self.__class__, self).setUp()

    def get_sarsa_trainer(self, environment):
        return self.get_sarsa_trainer_reward_boost(environment, {})

    def get_sarsa_trainer_reward_boost(self, environment, reward_shape):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.05,
            reward_burnin=10,
            maxq_learning=False,
            reward_boost=reward_shape,
        )
        training_parameters = TrainingParameters(
            layers=[-1, -1],
            activations=["linear"],
            minibatch_size=self.minibatch_size,
            learning_rate=0.05,
            optimizer="ADAM",
        )
        return DQNTrainer(
            DiscreteActionModelParameters(
                actions=environment.ACTIONS,
                rl=rl_parameters,
                training=training_parameters,
            ),
            environment.normalization,
        )

    def test_trainer_maxq(self):
        environment = Gridworld()
        maxq_sarsa_parameters = DiscreteActionModelParameters(
            actions=environment.ACTIONS,
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=True,
            ),
            training=TrainingParameters(
                layers=[-1, 1],
                activations=["linear"],
                minibatch_size=self.minibatch_size,
                learning_rate=0.01,
                optimizer="ADAM",
            ),
        )
        # construct the new trainer that using maxq
        maxq_trainer = DQNTrainer(maxq_sarsa_parameters, environment.normalization)

        samples = environment.generate_samples(200000, 0.5)
        predictor = maxq_trainer.predictor()
        tdps = environment.preprocess_samples(samples, self.minibatch_size)
        evaluator = GridworldEvaluator(environment, True, DISCOUNT)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertGreater(evaluator.mc_loss[-1], 0.3)

        for _ in range(5):
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                maxq_trainer.train(tdp)

        predictor = maxq_trainer.predictor()
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_trainer_sarsa_enum(self):
        environment = GridworldEnum()
        samples = environment.generate_samples(200000, 1.0)
        evaluator = GridworldEvaluator(environment, False, DISCOUNT)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertGreater(evaluator.mc_loss[-1], 0.12)

        for _ in range(5):
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_evaluator_ground_truth(self):
        environment = Gridworld()
        samples = environment.generate_samples(200000, 1.0)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.reward_timelines = []
        for tv in true_values:
            samples.reward_timelines.append({0: tv})
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(environment.ACTIONS, 1, DISCOUNT)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for _ in range(5):
            first = True
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                trainer.train(tdp, evaluator if first else None)
                first = False

        self.assertLess(evaluator.td_loss[-1], 0.01)
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_evaluator_timeline(self):
        environment = Gridworld()
        samples = environment.generate_samples(200000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(environment.ACTIONS, 1, DISCOUNT)

        tdps = environment.preprocess_samples(samples, self.minibatch_size)
        for _ in range(5):
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                trainer.train(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.2)
        self.assertLess(evaluator.mc_loss[-1], 0.2)

    def test_reward_boost(self):
        environment = Gridworld()
        reward_boost = {"L": 100, "R": 200, "U": 300, "D": 400}
        trainer = self.get_sarsa_trainer_reward_boost(environment, reward_boost)
        predictor = trainer.predictor()
        samples = environment.generate_samples(200000, 1.0)
        rewards_update = []
        for action, reward in zip(samples.actions, samples.rewards):
            rewards_update.append(reward - reward_boost[action])
        samples.rewards = rewards_update
        evaluator = GridworldEvaluator(environment, False, DISCOUNT)

        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertGreater(evaluator.mc_loss[-1], 0.12)

        for _ in range(5):
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                trainer.train(tdp, None)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)
