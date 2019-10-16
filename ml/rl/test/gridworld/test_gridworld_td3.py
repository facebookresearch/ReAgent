#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import numpy as np
import numpy.testing as npt
import torch
from ml.rl.models.actor import FullyConnectedActor
from ml.rl.models.output_transformer import ActorOutputTransformer
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.parameters import (
    FeedForwardParameters,
    OptimizerParameters,
    RLParameters,
    TD3ModelParameters,
    TD3TrainingParameters,
)
from ml.rl.prediction.dqn_torch_predictor import ParametricDqnTorchPredictor
from ml.rl.prediction.predictor_wrapper import (
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import (
    get_num_output_features,
    sort_features_by_normalization,
)
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.training.rl_exporter import ActorExporter
from ml.rl.training.td3_trainer import TD3Trainer


class TestGridworldTD3(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 4096
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_td3_parameters(self, use_2_q_functions=False):
        return TD3ModelParameters(
            rl=RLParameters(gamma=DISCOUNT, target_update_rate=0.01),
            training=TD3TrainingParameters(
                minibatch_size=self.minibatch_size,
                use_2_q_functions=use_2_q_functions,
                q_network_optimizer=OptimizerParameters(),
                actor_network_optimizer=OptimizerParameters(),
            ),
            q_network=FeedForwardParameters(
                layers=[128, 64], activations=["relu", "relu"]
            ),
            actor_network=FeedForwardParameters(
                layers=[128, 64], activations=["relu", "relu"]
            ),
        )

    def get_td3_trainer(self, env, parameters, use_gpu):
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
        actor_network = FullyConnectedActor(
            state_dim,
            action_dim,
            parameters.actor_network.layers,
            parameters.actor_network.activations,
        )

        if use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            actor_network.cuda()

        return TD3Trainer(
            q1_network,
            actor_network,
            parameters,
            q2_network=q2_network,
            use_gpu=use_gpu,
        )

    def get_predictor(self, trainer, environment):
        state_preprocessor = Preprocessor(environment.normalization, False)
        action_preprocessor = Preprocessor(environment.normalization_action, False)
        q_network = self.current_predictor_network
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, action_preprocessor
        )
        serving_module = ParametricDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor
        )
        predictor = ParametricDqnTorchPredictor(serving_module)
        return predictor

    def get_actor_predictor(self, trainer, environment):
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization
        )

        output_transformer = ActorOutputTransformer(
            sort_features_by_normalization(environment.normalization_action)[0],
            environment.max_action_range.reshape(-1),
            environment.min_action_range.reshape(-1),
        )

        predictor = ActorExporter(
            trainer.actor_network, feature_extractor, output_transformer
        ).export()
        return predictor

    def _test_td3_trainer(self, use_gpu=False, **kwargs):
        environment = GridworldContinuous()
        trainer = self.get_td3_trainer(
            environment, self.get_td3_parameters(**kwargs), use_gpu
        )
        evaluator = GridworldContinuousEvaluator(
            environment, assume_optimal_policy=False, gamma=DISCOUNT
        )

        self.current_predictor_network = trainer.q1_network
        self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

        if trainer.q2_network is not None:
            self.current_predictor_network = trainer.q2_network
            self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

        # Make sure actor predictor works
        actor_predictor = self.get_actor_predictor(trainer, environment)
        preds = actor_predictor.predict(evaluator.logged_states)
        self._test_save_load_actor(preds, actor_predictor, evaluator.logged_states)

    def _test_save_load_actor(
        self, before_preds, predictor, states, dbtype="minidb", check_equality=False
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            predictor.save(tmp_path, dbtype)
            new_predictor = type(predictor).load(tmp_path, dbtype)
            after_preds = new_predictor.predict(states)
        if check_equality:
            self._check_output_match(before_preds, after_preds)
        else:
            # Check if dims match
            self.assertEqual(len(before_preds), len(after_preds))

    def _check_output_match(self, a_preds, b_preds):
        self.assertEqual(len(a_preds), len(b_preds))
        self.assertEqual(a_preds[0].keys(), b_preds[0].keys())
        keys = list(a_preds[0].keys())

        a_array = [[r[k] for k in keys] for r in a_preds]
        b_array = [[r[k] for k in keys] for r in b_preds]
        npt.assert_allclose(a_array, b_array)

    def test_td3_trainer(self):
        self._test_td3_trainer()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_td3_trainer_gpu(self):
        self._test_td3_trainer(use_gpu=True)

    def test_td3_trainer_use_2_q_functions(self):
        self._test_td3_trainer(use_2_q_functions=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_td3_trainer_gpu_use_2_q_functions(self):
        self._test_td3_trainer(use_2_q_functions=True, use_gpu=True)
