#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest
from copy import deepcopy

import numpy as np
import torch
from ml.rl.models.actor import ActorWithPreprocessing, GaussianFullyConnectedActor
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.models.output_transformer import (
    ActorOutputTransformer,
    ParametricActionOutputTransformer,
)
from ml.rl.models.parametric_dqn import (
    FullyConnectedParametricDQN,
    ParametricDQNWithPreprocessing,
)
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import (
    get_num_output_features,
    sort_features_by_normalization,
)
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import (
    GridworldContinuousEvaluator,
    GridworldDDPGEvaluator,
)
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.thrift.core.ttypes import (
    FeedForwardParameters,
    OptimizerParameters,
    RLParameters,
    SACModelParameters,
    SACTrainingParameters,
)
from ml.rl.training.rl_exporter import ParametricDQNExporter
from ml.rl.training.sac_trainer import SACTrainer


class TestGridworldSAC(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 4096
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super(self.__class__, self).setUp()

    def get_sac_parameters(self, use_2_q_functions=False):
        return SACModelParameters(
            rl=RLParameters(gamma=DISCOUNT, target_update_rate=0.5, reward_burnin=100),
            training=SACTrainingParameters(
                minibatch_size=self.minibatch_size,
                use_2_q_functions=use_2_q_functions,
                q_network_optimizer=OptimizerParameters(),
                value_network_optimizer=OptimizerParameters(),
                actor_network_optimizer=OptimizerParameters(),
            ),
            q_network=FeedForwardParameters(
                layers=[128, 64], activations=["relu", "relu"]
            ),
            value_network=FeedForwardParameters(
                layers=[128, 64], activations=["relu", "relu"]
            ),
            actor_network=FeedForwardParameters(
                layers=[128, 64], activations=["relu", "relu"]
            ),
        )

    def get_sac_trainer(self, env, parameters, use_gpu):
        state_dim = get_num_output_features(env.normalization)
        action_dim = get_num_output_features(env.normalization_action)
        q1_network = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            parameters.q_network.layers,
            parameters.q_network.activations,
        )
        q2_network = None
        if parameters.training.use_2_q_functions:
            q2_network = FullyConnectedParametricDQN(
                state_dim,
                action_dim,
                parameters.q_network.layers,
                parameters.q_network.activations,
            )
        value_network = FullyConnectedNetwork(
            [state_dim] + parameters.value_network.layers + [1],
            parameters.value_network.activations + ["linear"],
        )
        actor_network = GaussianFullyConnectedActor(
            state_dim,
            action_dim,
            parameters.actor_network.layers,
            parameters.actor_network.activations,
        )
        if use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            value_network.cuda()
            actor_network.cuda()

        value_network_target = deepcopy(value_network)
        return SACTrainer(
            q1_network,
            value_network,
            value_network_target,
            actor_network,
            parameters,
            q2_network=q2_network,
        )

    def get_critic_exporter(self, trainer, environment):
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization,
            action_normalization_parameters=environment.normalization_action,
        )
        output_transformer = ParametricActionOutputTransformer()

        return ParametricDQNExporter(
            trainer.q1_network,
            feature_extractor,
            output_transformer,
            Preprocessor(environment.normalization, False, True),
            Preprocessor(environment.normalization_action, False, True),
        )

    def get_actor_predictor(self, trainer, environment):
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization
        )

        output_transformer = ActorOutputTransformer(
            sort_features_by_normalization(environment.normalization_action)[0],
            environment.max_action_range.reshape(-1),
            environment.min_action_range.reshape(-1),
        )

        def container(actor_network):
            return ActorWithPreprocessing(
                actor_network, Preprocessor(environment.normalization, False, True)
            )

        return trainer.actor_predictor(feature_extractor, output_transformer, container)

    def _test_sac_trainer(self, use_2_q_functions=False, use_gpu=False):
        environment = GridworldContinuous()
        trainer = self.get_sac_trainer(
            environment, self.get_sac_parameters(use_2_q_functions), use_gpu
        )
        evaluator = GridworldContinuousEvaluator(
            environment,
            assume_optimal_policy=False,
            gamma=DISCOUNT,
            use_int_features=False,
        )

        exporter = self.get_critic_exporter(trainer, environment)

        self.test_save_load = False
        self.tolerance_threshold = 0.12
        if use_gpu:
            self.run_pre_training_eval = False
        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

        # Make sure actor predictor works
        actor_predictor = self.get_actor_predictor(trainer, environment)
        # Just test that it doesn't blow up
        actor_predictor.predict(evaluator.logged_states, None)
        # TODO: run predictor and check results

    def test_sac_trainer(self):
        self._test_sac_trainer()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_gpu(self):
        self._test_sac_trainer(use_gpu=True)

    def test_sac_trainer_use_2_q_functions(self):
        self._test_sac_trainer(use_2_q_functions=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_gpu_use_2_q_functions(self):
        self._test_sac_trainer(use_2_q_functions=True, use_gpu=True)
