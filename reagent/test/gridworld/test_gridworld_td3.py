#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import numpy as np
import numpy.testing as npt
import torch
from reagent.models.actor import FullyConnectedActor
from reagent.models.parametric_dqn import FullyConnectedParametricDQN
from reagent.parameters import (
    FeedForwardParameters,
    OptimizerParameters,
    RLParameters,
    TD3ModelParameters,
    TD3TrainingParameters,
)
from reagent.prediction.dqn_torch_predictor import (
    ActorTorchPredictor,
    ParametricDqnTorchPredictor,
)
from reagent.prediction.predictor_wrapper import (
    ActorPredictorWrapper,
    ActorWithPreprocessor,
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from reagent.preprocessing.normalization import (
    get_num_output_features,
    sort_features_by_normalization,
)
from reagent.preprocessing.postprocessor import Postprocessor
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.test.gridworld.gridworld_base import DISCOUNT
from reagent.test.gridworld.gridworld_continuous import GridworldContinuous
from reagent.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from reagent.test.gridworld.gridworld_test_base import GridworldTestBase
from reagent.training.td3_trainer import TD3Trainer


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
        state_preprocessor = Preprocessor(environment.normalization, False)
        postprocessor = Postprocessor(
            environment.normalization_continuous_action, False
        )
        actor_with_preprocessor = ActorWithPreprocessor(
            trainer.actor_network.cpu_model().eval(), state_preprocessor, postprocessor
        )
        serving_module = ActorPredictorWrapper(actor_with_preprocessor)
        predictor = ActorTorchPredictor(
            serving_module,
            sort_features_by_normalization(environment.normalization_continuous_action)[
                0
            ],
        )
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
        self, before_preds, predictor, states, check_equality=False
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            torch.jit.save(predictor.model, tmp_path)
            loaded_model = torch.jit.load(tmp_path)
            new_predictor = type(predictor)(loaded_model, predictor.action_feature_ids)
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
