#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import numpy as np
import numpy.testing as npt
import torch
from ml.rl.models.actor import DirichletFullyConnectedActor, GaussianFullyConnectedActor
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.parameters import FeedForwardParameters, OptimizerParameters, RLParameters
from ml.rl.prediction.dqn_torch_predictor import (
    ActorTorchPredictor,
    ParametricDqnTorchPredictor,
)
from ml.rl.prediction.predictor_wrapper import (
    ActorPredictorWrapper,
    ActorWithPreprocessor,
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from ml.rl.preprocessing.normalization import (
    get_num_output_features,
    sort_features_by_normalization,
)
from ml.rl.preprocessing.postprocessor import Postprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.training.sac_trainer import SACTrainer, SACTrainerParameters


class TestGridworldSAC(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 4096
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_sac_trainer(
        self,
        env,
        use_gpu,
        use_2_q_functions=False,
        logged_action_uniform_prior=True,
        constrain_action_sum=False,
        use_value_network=True,
    ):
        q_network_params = FeedForwardParameters(
            layers=[128, 64], activations=["relu", "relu"]
        )
        value_network_params = FeedForwardParameters(
            layers=[128, 64], activations=["relu", "relu"]
        )
        actor_network_params = FeedForwardParameters(
            layers=[128, 64], activations=["relu", "relu"]
        )

        state_dim = get_num_output_features(env.normalization)
        action_dim = get_num_output_features(env.normalization_continuous_action)
        q1_network = FullyConnectedParametricDQN(
            state_dim, action_dim, q_network_params.layers, q_network_params.activations
        )
        q2_network = None
        if use_2_q_functions:
            q2_network = FullyConnectedParametricDQN(
                state_dim,
                action_dim,
                q_network_params.layers,
                q_network_params.activations,
            )
        if constrain_action_sum:
            actor_network = DirichletFullyConnectedActor(
                state_dim,
                action_dim,
                actor_network_params.layers,
                actor_network_params.activations,
            )
        else:
            actor_network = GaussianFullyConnectedActor(
                state_dim,
                action_dim,
                actor_network_params.layers,
                actor_network_params.activations,
            )

        value_network = None
        if use_value_network:
            value_network = FullyConnectedNetwork(
                [state_dim] + value_network_params.layers + [1],
                value_network_params.activations + ["linear"],
            )

        if use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            if value_network:
                value_network.cuda()
            actor_network.cuda()

        parameters = SACTrainerParameters(
            rl=RLParameters(gamma=DISCOUNT, target_update_rate=0.5),
            minibatch_size=self.minibatch_size,
            q_network_optimizer=OptimizerParameters(),
            value_network_optimizer=OptimizerParameters(),
            actor_network_optimizer=OptimizerParameters(),
            alpha_optimizer=OptimizerParameters(),
            logged_action_uniform_prior=logged_action_uniform_prior,
        )

        return SACTrainer(
            q1_network,
            actor_network,
            parameters,
            use_gpu=use_gpu,
            value_network=value_network,
            q2_network=q2_network,
        )

    def get_predictor(self, trainer, environment):
        state_preprocessor = Preprocessor(environment.normalization, False)
        action_preprocessor = Preprocessor(
            environment.normalization_continuous_action, False
        )
        q_network = trainer.q1_network
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

    def _test_sac_trainer(self, use_gpu=False, **kwargs):
        environment = GridworldContinuous()
        trainer = self.get_sac_trainer(environment, use_gpu, **kwargs)
        evaluator = GridworldContinuousEvaluator(
            environment, assume_optimal_policy=False, gamma=DISCOUNT
        )

        self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

        # Make sure actor predictor works
        actor_predictor = self.get_actor_predictor(trainer, environment)
        # Just test that it doesn't blow up
        preds = actor_predictor.predict(evaluator.logged_states)
        self._test_save_load_actor(preds, actor_predictor, evaluator.logged_states)
        # TODO: run predictor and check results

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
            # Check if dims match for stochastic outputs in SAC
            self.assertEqual(len(before_preds), len(after_preds))

    def _check_output_match(self, a_preds, b_preds):
        self.assertEqual(len(a_preds), len(b_preds))
        self.assertEqual(a_preds[0].keys(), b_preds[0].keys())
        keys = list(a_preds[0].keys())

        a_array = [[r[k] for k in keys] for r in a_preds]
        b_array = [[r[k] for k in keys] for r in b_preds]
        npt.assert_allclose(a_array, b_array)

    def test_sac_trainer(self):
        self._test_sac_trainer()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_gpu(self):
        self._test_sac_trainer(use_gpu=True)

    def test_sac_trainer_not_use_value_network(self):
        self._test_sac_trainer(use_value_network=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_not_use_value_network_gpu(self):
        self._test_sac_trainer(use_value_network=False, use_gpu=True)

    def test_sac_trainer_use_2_q_functions(self):
        self._test_sac_trainer(use_2_q_functions=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_gpu_use_2_q_functions(self):
        self._test_sac_trainer(use_2_q_functions=True, use_gpu=True)

    def test_sac_trainer_model_propensity(self):
        self._test_sac_trainer(logged_action_uniform_prior=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_model_propensity_gpu(self):
        self._test_sac_trainer(use_gpu=True, logged_action_uniform_prior=True)

    def test_sac_trainer_w_dirichlet_actor(self):
        self._test_sac_trainer(constrain_action_sum=True)

    # TODO: Renable when PyTorch supports backwards pass in CUDA.
    @unittest.skipIf(True or not torch.cuda.is_available(), "CUDA not available")
    def test_sac_trainer_w_dirichlet_actor_gpu(self):
        self._test_sac_trainer(use_gpu=True, constrain_action_sum=True)
