#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest

import numpy as np
import torch
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.parameters import (
    ContinuousActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.prediction.dqn_torch_predictor import ParametricDqnTorchPredictor
from ml.rl.prediction.predictor_wrapper import (
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


class TestGridworldParametric(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 512
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_sarsa_parameters(self):
        return ContinuousActionModelParameters(
            rl=RLParameters(
                gamma=DISCOUNT, target_update_rate=1.0, maxq_learning=False
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
        )

    def get_trainer(
        self, environment, parameters=None, use_gpu=False, use_all_avail_gpus=False
    ):
        parameters = parameters or self.get_sarsa_parameters()
        q_network = FullyConnectedParametricDQN(
            state_dim=get_num_output_features(environment.normalization),
            action_dim=get_num_output_features(environment.normalization_action),
            sizes=parameters.training.layers[1:-1],
            activations=parameters.training.activations[:-1],
        )
        reward_network = FullyConnectedParametricDQN(
            state_dim=get_num_output_features(environment.normalization),
            action_dim=get_num_output_features(environment.normalization_action),
            sizes=parameters.training.layers[1:-1],
            activations=parameters.training.activations[:-1],
        )
        if use_gpu:
            q_network = q_network.cuda()
            reward_network = reward_network.cuda()
            if use_all_avail_gpus:
                q_network = q_network.get_distributed_data_parallel_model()
                reward_network = reward_network.get_distributed_data_parallel_model()

        q_network_target = q_network.get_target_network()
        trainer = ParametricDQNTrainer(
            q_network, q_network_target, reward_network, parameters
        )
        return trainer

    def get_predictor(self, trainer, environment):
        state_preprocessor = Preprocessor(environment.normalization, False)
        action_preprocessor = Preprocessor(environment.normalization_action, False)
        q_network = trainer.q_network
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, action_preprocessor
        )
        serving_module = ParametricDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor
        )
        predictor = ParametricDqnTorchPredictor(serving_module)
        return predictor

    def _test_trainer_sarsa(self, use_gpu=False, use_all_avail_gpus=False):
        environment = GridworldContinuous()
        evaluator = GridworldContinuousEvaluator(
            environment, assume_optimal_policy=False, gamma=DISCOUNT
        )

        trainer = self.get_trainer(environment, None, use_gpu, use_all_avail_gpus)

        self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

    def test_modular_trainer_sarsa(self):
        self._test_trainer_sarsa()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_modular_trainer_sarsa_gpu(self):
        self._test_trainer_sarsa(use_gpu=True)
