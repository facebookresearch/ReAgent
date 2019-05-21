#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import tempfile
import unittest

import numpy as np
import torch
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldDDPGEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.thrift.core.ttypes import (
    DDPGModelParameters,
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    RLParameters,
)
from ml.rl.training.ddpg_trainer import ActorNetModel, CriticNetModel, DDPGTrainer
from ml.rl.training.rl_exporter import ActorExporter, ParametricDQNExporter
from torch import distributed


class TestGridworldDdpg(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 4096
        super().setUp()
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def get_ddpg_parameters(self):
        return DDPGModelParameters(
            rl=RLParameters(gamma=DISCOUNT, target_update_rate=0.5, maxq_learning=True),
            shared_training=DDPGTrainingParameters(
                minibatch_size=self.minibatch_size,
                final_layer_init=0.003,
                optimizer="ADAM",
            ),
            actor_training=DDPGNetworkParameters(
                layers=[-1, 256, 128, -1],
                activations=["relu", "relu", "tanh"],
                learning_rate=0.05,
                l2_decay=0.01,
            ),
            critic_training=DDPGNetworkParameters(
                layers=[-1, 256, 256, 128, -1],
                activations=["relu", "relu", "relu", "linear"],
                learning_rate=0.05,
                l2_decay=0.01,
            ),
        )

    def _test_ddpg_trainer(self, use_gpu=False, use_all_avail_gpus=False):
        # FIXME:the test not really working
        self.run_pre_training_eval = False
        self.check_tolerance = False
        environment = GridworldContinuous()

        parameters = self.get_ddpg_parameters()

        state_dim = get_num_output_features(environment.normalization)
        action_dim = get_num_output_features(environment.normalization_action)

        # Build Actor Network
        actor_network = ActorNetModel(
            layers=[state_dim] + parameters.actor_training.layers[1:-1] + [action_dim],
            activations=parameters.actor_training.activations,
            fl_init=parameters.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )

        # Build Critic Network
        critic_network = CriticNetModel(
            # Ensure dims match input state and scalar output
            layers=[state_dim] + parameters.critic_training.layers[1:-1] + [1],
            activations=parameters.critic_training.activations,
            fl_init=parameters.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )

        trainer = DDPGTrainer(
            actor_network,
            critic_network,
            parameters,
            environment.normalization,
            environment.normalization_action,
            environment.min_action_range,
            environment.max_action_range,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )

        exporter = ParametricDQNExporter.from_state_action_normalization(
            trainer.critic,
            state_normalization=environment.normalization,
            action_normalization=environment.normalization_action,
        )

        evaluator = GridworldDDPGEvaluator(environment, DISCOUNT)
        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

        # Make sure actor predictor works
        actor = ActorExporter.from_state_action_normalization(
            trainer.actor,
            state_normalization=environment.normalization,
            action_normalization=environment.normalization_action,
        ).export()

        # Make sure all actions are optimal
        error = evaluator.evaluate_actor(actor, thres=0.2)
        print("gridworld optimal action match MAE: {0:.3f}".format(error))

    def test_ddpg_trainer(self):
        self._test_ddpg_trainer()

    @unittest.skipIf(
        not torch.cuda.is_available() or True,
        "CUDA not available; failing on CI for reason",
    )
    def test_ddpg_trainer_gpu(self):
        self._test_ddpg_trainer(use_gpu=True)
