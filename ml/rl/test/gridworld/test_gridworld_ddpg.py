#!/usr/bin/env python3

import random
import numpy as np
import unittest

from ml.rl.thrift.core.ttypes import (
    RLParameters,
    DDPGModelParameters,
    DDPGTrainingParameters,
    DDPGNetworkParameters,
)
from ml.rl.training.ddpg_trainer import DDPGTrainer
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldDDPGEvaluator


class TestGridworldContinuous(unittest.TestCase):
    def setUp(self):
        self.minibatch_size = 1024
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

    def get_ddpg_parameters(self):
        return DDPGModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=False,
            ),
            shared_training=DDPGTrainingParameters(
                minibatch_size=self.minibatch_size,
                final_layer_init=0.003,
                optimizer="ADAM",
            ),
            actor_training=DDPGNetworkParameters(
                layers=[-1, 400, 300, -1],
                activations=["relu", "relu", "tanh"],
                learning_rate=0.0001,
            ),
            critic_training=DDPGNetworkParameters(
                layers=[-1, 400, 300, -1],
                activations=["relu", "relu", "tanh"],
                learning_rate=0.001,
                l2_decay=0.01,
            ),
        )

    def test_ddpg_trainer(self):
        environment = GridworldContinuous()
        samples = environment.generate_samples(200000, 1.0)
        epochs = 3
        trainer = DDPGTrainer(
            self.get_ddpg_parameters(),
            environment.normalization,
            environment.normalization_action,
        )
        evaluator = GridworldDDPGEvaluator(environment, True)
        tdps = environment.preprocess_samples(samples, self.minibatch_size)

        for epoch in range(epochs):
            print("On epoch {} of {}".format(epoch + 1, epochs))
            critic_predictor = trainer.predictor()
            evaluator.evaluate_critic(critic_predictor)
            for tdp in tdps:
                tdp.rewards = tdp.rewards.flatten()
                tdp.not_terminals = tdp.not_terminals.flatten()
                trainer.train(tdp)

        critic_predictor = trainer.predictor()
        error = evaluator.evaluate_critic(critic_predictor)
        print("gridworld MAE: {0:.3f}".format(error))
        # For now we are disabling this test until we can get DDPG to be healthy
        # on discrete action domains (T30810709).
        # assert error < 0.1, "gridworld MAE: {} > {}".format(error, 0.1)
