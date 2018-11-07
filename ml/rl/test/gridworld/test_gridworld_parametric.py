#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest

import numpy as np
import torch
from ml.rl.models.output_transformer import ParametricActionOutputTransformer
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.thrift_types import (
    ContinuousActionModelParameters,
    FactorizationParameters,
    FeedForwardParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training._parametric_dqn_trainer import _ParametricDQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.training.rl_exporter import ParametricDQNExporter


class TestGridworldParametric(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 512
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super(self.__class__, self).setUp()

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
                        layers=[-1, 128, 64], activations=["relu", "linear"]
                    ),
                    action=FeedForwardParameters(
                        layers=[-1, 128, 64], activations=["relu", "linear"]
                    ),
                ),
                minibatch_size=self.minibatch_size,
                learning_rate=0.03,
                optimizer="ADAM",
            ),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
            in_training_cpe=InTrainingCPEParameters(mdp_sampled_rate=0.1),
        )

    def get_sarsa_trainer_exporter(
        self, environment, parameters=None, use_gpu=False, use_all_avail_gpus=False
    ):
        parameters = parameters or self.get_sarsa_parameters()
        trainer = ParametricDQNTrainer(
            parameters,
            environment.normalization,
            environment.normalization_action,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        return (trainer, trainer)

    def get_modular_sarsa_trainer_exporter(
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
                q_network = q_network.get_data_parallel_model()
                reward_network = reward_network.get_data_parallel_model()

        q_network_target = q_network.get_target_network()
        trainer = _ParametricDQNTrainer(
            q_network, q_network_target, reward_network, parameters
        )
        state_preprocessor = Preprocessor(environment.normalization, False, True)
        action_preprocessor = Preprocessor(
            environment.normalization_action, False, True
        )
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization,
            action_normalization_parameters=environment.normalization_action,
        )
        output_transformer = ParametricActionOutputTransformer()
        exporter = ParametricDQNExporter(
            q_network,
            feature_extractor,
            output_transformer,
            state_preprocessor,
            action_preprocessor,
        )
        return (trainer, exporter)

    def _test_trainer_sarsa(
        self, use_gpu=False, use_all_avail_gpus=False, modular=False
    ):
        environment = GridworldContinuous()
        evaluator = GridworldContinuousEvaluator(
            environment,
            assume_optimal_policy=False,
            gamma=DISCOUNT,
            use_int_features=False,
        )

        if modular:
            # FIXME: the exporter should make a copy of the model; moving it to CPU inplace
            if use_gpu:
                self.run_pre_training_eval = False
            if use_all_avail_gpus:
                self.tolerance_threshold = 0.11
            trainer, exporter = self.get_modular_sarsa_trainer_exporter(
                environment, None, use_gpu, use_all_avail_gpus
            )
        else:
            trainer, exporter = self.get_sarsa_trainer_exporter(
                environment, None, use_gpu, use_all_avail_gpus
            )

        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

    def test_trainer_sarsa(self):
        self._test_trainer_sarsa()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_gpu(self):
        self._test_trainer_sarsa(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_all_gpus(self):
        self._test_trainer_sarsa(use_gpu=True, use_all_avail_gpus=True)

    def test_modular_trainer_sarsa(self):
        self._test_trainer_sarsa(modular=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_modular_trainer_sarsa_gpu(self):
        self._test_trainer_sarsa(use_gpu=True, modular=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_modular_trainer_sarsa_all_gpus(self):
        self._test_trainer_sarsa(use_gpu=True, use_all_avail_gpus=True, modular=True)

    def _test_trainer_sarsa_factorized(self, use_gpu=False, use_all_avail_gpus=False):
        self.check_tolerance = False
        self.tolerance_threshold = 0.15
        environment = GridworldContinuous()
        trainer, exporter = self.get_sarsa_trainer_exporter(
            environment,
            self.get_sarsa_parameters_factorized(),
            use_gpu,
            use_all_avail_gpus,
        )
        evaluator = GridworldContinuousEvaluator(environment, False, DISCOUNT, False)
        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

    def test_trainer_sarsa_factorized(self):
        self._test_trainer_sarsa_factorized()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_factorized_gpu(self):
        self._test_trainer_sarsa_factorized(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_factorized_all_gpus(self):
        self._test_trainer_sarsa_factorized(use_gpu=True, use_all_avail_gpus=True)
