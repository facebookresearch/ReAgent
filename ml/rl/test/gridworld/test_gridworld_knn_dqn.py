#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.thrift.core.ttypes import (
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    KNNDQNModelParameters,
    RLParameters,
)
from ml.rl.training.knn_dqn_trainer import KNNDQNTrainer


class TestGridworldContinuous(unittest.TestCase):
    def setUp(self):
        self.minibatch_size = 1024
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def get_parameters(self, env, action_dim=4, k=2):
        # ONNX does't export `normalize`
        return KNNDQNModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=100,
                maxq_learning=False,
            ),
            shared_training=DDPGTrainingParameters(
                minibatch_size=self.minibatch_size,
                final_layer_init=0.003,
                optimizer="ADAM",
            ),
            actor_training=DDPGNetworkParameters(
                layers=[-1, 64, 32, -1],
                activations=["relu", "relu", "tanh"],
                learning_rate=0.1,
            ),
            critic_training=DDPGNetworkParameters(
                layers=[-1, 64, 32, -1],
                activations=["relu", "relu", "linear"],
                learning_rate=0.1,
                l2_decay=0.999,
            ),
            num_actions=len(env.ACTIONS),
            action_dim=action_dim,
            k=k,
        )

    def test_knn_dqn_trainer(self):
        environment = Gridworld()
        samples = environment.generate_samples(200000, 1.0)
        evaluator = GridworldEvaluator(environment, False, DISCOUNT, False, samples)

        parameters = self.get_parameters(environment)
        trainer = KNNDQNTrainer(parameters, environment.normalization)

        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, one_hot_action=False
        )

        predictor = trainer.predictor(environment.ACTIONS)

        evaluator.evaluate(predictor)
        print(
            "Pre-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        pre_train_loss = evaluator.mc_loss[-1]
        pre_train_value = evaluator.value_doubly_robust[-1]
        self.assertGreater(evaluator.mc_loss[-1], 0.09)

        for tdp in tdps:
            tdp.rewards = tdp.rewards.flatten()
            tdp.not_terminals = tdp.not_terminals.flatten()
            trainer.train(tdp)

        predictor = trainer.predictor(environment.ACTIONS)
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.value_doubly_robust[-1],
        )
        print("Pre-Training eval: ", pre_train_loss, pre_train_value)
        self.assertLess(evaluator.mc_loss[-1], pre_train_loss)
