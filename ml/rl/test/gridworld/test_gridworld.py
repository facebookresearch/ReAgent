#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_enum import GridworldEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.evaluator import Evaluator


class TestGridworld(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        self.minibatch_size = 1024
        super(self.__class__, self).setUp()

    def get_sarsa_trainer(self, environment):
        return self.get_sarsa_trainer_reward_boost(environment, {})

    def get_sarsa_trainer_reward_boost(self, environment, reward_shape):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=1.0,
            reward_burnin=10,
            maxq_learning=False,
            reward_boost=reward_shape,
        )
        training_parameters = TrainingParameters(
            layers=[-1, -1],
            activations=["linear"],
            minibatch_size=self.minibatch_size,
            learning_rate=0.125,
            optimizer="ADAM",
        )
        return DiscreteActionTrainer(
            DiscreteActionModelParameters(
                actions=environment.ACTIONS,
                rl=rl_parameters,
                training=training_parameters,
                rainbow=RainbowDQNParameters(
                    double_q_learning=True, dueling_architecture=False
                ),
                in_training_cpe=InTrainingCPEParameters(mdp_sampled_rate=0.1),
            ),
            environment.normalization,
        )

    def test_trainer_sarsa(self):
        environment = Gridworld()
        samples = environment.generate_samples(150000, 1.0)
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )

        for _ in range(2):
            for tdp in tdps:
                trainer.train_numpy(tdp, None)
            evaluator.evaluate(predictor)

        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_trainer_sarsa_enum(self):
        environment = GridworldEnum()
        samples = environment.generate_samples(150000, 1.0)
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )

        for _ in range(2):
            for tdp in tdps:
                trainer.train_numpy(tdp, None)
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
        evaluator = Evaluator(environment.ACTIONS, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for _ in range(2):
            for tdp in tdps:
                trainer.train_numpy(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_reward_boost(self):
        environment = Gridworld()
        reward_boost = {"L": 100, "R": 200, "U": 300, "D": 400}
        trainer = self.get_sarsa_trainer_reward_boost(environment, reward_boost)
        predictor = trainer.predictor()
        samples = environment.generate_samples(150000, 1.0)
        rewards_update = []
        for action, reward in zip(samples.actions, samples.rewards):
            rewards_update.append(reward - reward_boost[action])
        samples.rewards = rewards_update
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)

        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )

        for _ in range(2):
            for tdp in tdps:
                trainer.train_numpy(tdp, None)

        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)
