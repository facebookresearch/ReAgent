#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.cb_fully_connected_network import CBFullyConnectedNetwork
from reagent.training.cb.supervised_trainer import SupervisedTrainer
from reagent.training.parameters import SupervisedTrainerParameters


class TestSupervisedTrainer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2

        self.num_arms = 2
        self.params = SupervisedTrainerParameters()

        self.x_dim = 5
        policy_network = CBFullyConnectedNetwork(self.x_dim, [8, 10])
        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())

        self.trainer = SupervisedTrainer(self.policy, **self.params.asdict())
        self.batch = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2, 3, 6, 7],
                        [1, 2, 3, 10, 11],
                    ],
                    [
                        [1, 4, 5, 8, 9],
                        [1, 4, 5, 12, 13],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
        )

    def test_supervised_training_step(self):
        self.trainer.training_step(self.batch, 0)
        self.trainer.on_train_epoch_end()

    def test_linucb_training_multiple_epochs(self):
        obss = []
        for i in range(self.batch_size):
            obss.append(
                CBInput(
                    context_arm_features=self.batch.context_arm_features[
                        i : i + 1, :, :
                    ],
                    action=self.batch.action[[i]],
                    reward=self.batch.reward[[i]],
                )
            )

        scorer = CBFullyConnectedNetwork(self.x_dim, [7, 9])
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = SupervisedTrainer(policy)

        trainer.training_step(obss[0], 0)
        trainer.on_train_epoch_end()
        trainer.training_step(obss[1], 1)
        trainer.on_train_epoch_end()
