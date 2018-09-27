#!/usr/bin/env python3

import os
import random
import tempfile
import unittest

import numpy as np
import torch
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT, Samples
from ml.rl.test.gridworld.gridworld_enum import GridworldEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.evaluator import Evaluator


class TestGridworld(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        self.minibatch_size = 4096
        super(self.__class__, self).setUp()

    def get_sarsa_trainer(self, environment, dueling):
        return self.get_sarsa_trainer_reward_boost(environment, {}, dueling)

    def get_sarsa_trainer_reward_boost(self, environment, reward_shape, dueling):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=1.0,
            reward_burnin=10,
            maxq_learning=False,
            reward_boost=reward_shape,
        )
        training_parameters = TrainingParameters(
            layers=[-1, 128, -1] if dueling else [-1, -1],
            activations=["relu", "linear"] if dueling else ["linear"],
            minibatch_size=self.minibatch_size,
            learning_rate=0.125,
            optimizer="ADAM",
        )
        return DQNTrainer(
            DiscreteActionModelParameters(
                actions=environment.ACTIONS,
                rl=rl_parameters,
                training=training_parameters,
                rainbow=RainbowDQNParameters(
                    double_q_learning=True, dueling_architecture=dueling
                ),
                in_training_cpe=InTrainingCPEParameters(mdp_sampled_rate=0.1),
            ),
            environment.normalization,
        )

    def test_trainer_sarsa_enum(self):
        environment = GridworldEnum()
        samples = environment.generate_samples(500000, 1.0, DISCOUNT)
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)
        trainer = self.get_sarsa_trainer(environment, False)
        predictor = trainer.predictor()
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertGreater(evaluator.mc_loss[-1], 0.12)

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_evaluator_ground_truth_no_dueling(self):
        environment = Gridworld()
        samples = environment.generate_samples(500000, 1.0, DISCOUNT)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = true_values
        trainer = self.get_sarsa_trainer(environment, False)
        evaluator = Evaluator(environment.ACTIONS, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_evaluator_ground_truth_dueling(self):
        environment = Gridworld()
        samples = environment.generate_samples(500000, 1.0, DISCOUNT)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = true_values
        trainer = self.get_sarsa_trainer(environment, True)
        evaluator = Evaluator(environment.ACTIONS, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_reward_boost(self):
        environment = Gridworld()
        reward_boost = {"L": 100, "R": 200, "U": 300, "D": 400}
        trainer = self.get_sarsa_trainer_reward_boost(environment, reward_boost, False)
        predictor = trainer.predictor()
        samples = environment.generate_samples(500000, 1.0, DISCOUNT)
        rewards_update = []
        for action, reward in zip(samples.actions, samples.rewards):
            rewards_update.append(reward - reward_boost[action])
        samples.rewards = rewards_update
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)

        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            predictor.save(tmp_path, "minidb")
            new_predictor = DQNPredictor.load(tmp_path, "minidb", False)
            evaluator.evaluate(new_predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertGreater(evaluator.mc_loss[-1], 0.12)

        for tdp in tdps:
            trainer.train(tdp, None)

        predictor = trainer.predictor()
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            predictor.save(tmp_path, "minidb")
            new_predictor = DQNPredictor.load(tmp_path, "minidb", False)
            evaluator.evaluate(new_predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_predictor_export(self):
        """Verify that q-values before model export equal q-values after
        model export. Meant to catch issues with export logic."""
        environment = Gridworld()
        trainer = trainer = self.get_sarsa_trainer(environment, False)

        samples = Samples(
            mdp_ids=["0"],
            sequence_numbers=[0],
            states=[{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 15: 1.0, 24: 1.0}],
            actions=["D"],
            action_probabilities=[.5],
            rewards=[0],
            possible_actions=[["R", "D"]],
            next_states=[{5: 1.0}],
            next_actions=["U"],
            terminals=[False],
            possible_next_actions=[["R", "U", "D"]],
            episode_values=[0.00161],
        )
        tdps = environment.preprocess_samples(samples, 1)

        pre_export_q_values = trainer.q_network(tdps[0].states).detach().numpy()

        predictor = trainer.predictor()
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            predictor.save(tmp_path, "minidb")
            new_predictor = DQNPredictor.load(tmp_path, "minidb", False)

        post_export_q_values = new_predictor.predict([samples.states[0]])

        for i, action in enumerate(environment.ACTIONS):
            self.assertAlmostEquals(
                pre_export_q_values[0][i], post_export_q_values[0][action], places=4
            )
