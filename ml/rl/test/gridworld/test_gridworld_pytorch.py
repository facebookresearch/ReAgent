#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.models.output_transformer import DiscreteActionOutputTransformer
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT, Samples
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.rl_exporter import DQNExporter
from torch import distributed


class TestGridworld(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 512
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_sarsa_parameters(self, environment, reward_shape, dueling, clip_grad_norm):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=1.0,
            maxq_learning=False,
            reward_boost=reward_shape,
        )
        training_parameters = TrainingParameters(
            layers=[-1, 128, -1] if dueling else [-1, -1],
            activations=["relu", "linear"] if dueling else ["linear"],
            minibatch_size=self.minibatch_size,
            learning_rate=0.05,
            optimizer="ADAM",
            clip_grad_norm=clip_grad_norm,
        )
        return DiscreteActionModelParameters(
            actions=environment.ACTIONS,
            rl=rl_parameters,
            training=training_parameters,
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=dueling
            ),
        )

    def get_modular_sarsa_trainer(
        self,
        environment,
        dueling,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        return self.get_modular_sarsa_trainer_reward_boost(
            environment,
            {},
            dueling,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
        )

    def get_modular_sarsa_trainer_reward_boost(
        self,
        environment,
        reward_shape,
        dueling,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        parameters = self.get_sarsa_parameters(
            environment, reward_shape, dueling, clip_grad_norm
        )
        q_network = FullyConnectedDQN(
            state_dim=get_num_output_features(environment.normalization),
            action_dim=len(environment.ACTIONS),
            sizes=parameters.training.layers[1:-1],
            activations=parameters.training.activations[:-1],
        )
        reward_network = FullyConnectedDQN(
            state_dim=get_num_output_features(environment.normalization),
            action_dim=len(environment.ACTIONS),
            sizes=parameters.training.layers[1:-1],
            activations=parameters.training.activations[:-1],
        )
        q_network_cpe = FullyConnectedDQN(
            state_dim=get_num_output_features(environment.normalization),
            action_dim=len(environment.ACTIONS),
            sizes=parameters.training.layers[1:-1],
            activations=parameters.training.activations[:-1],
        )
        if use_gpu:
            q_network = q_network.cuda()
            reward_network = reward_network.cuda()
            q_network_cpe = q_network_cpe.cuda()
            if use_all_avail_gpus:
                q_network = q_network.get_distributed_data_parallel_model()
                reward_network = reward_network.get_distributed_data_parallel_model()
                q_network_cpe = q_network_cpe.get_distributed_data_parallel_model()

        q_network_target = q_network.get_target_network()
        q_network_cpe_target = q_network_cpe.get_target_network()
        trainer = DQNTrainer(
            q_network,
            q_network_target,
            reward_network,
            parameters,
            use_gpu,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
        )
        return trainer

    def get_modular_sarsa_trainer_exporter(
        self,
        environment,
        reward_shape,
        dueling,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        parameters = self.get_sarsa_parameters(
            environment, reward_shape, dueling, clip_grad_norm
        )
        trainer = self.get_modular_sarsa_trainer_reward_boost(
            environment,
            reward_shape,
            dueling,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
        )
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization
        )
        output_transformer = DiscreteActionOutputTransformer(parameters.actions)
        exporter = DQNExporter(trainer.q_network, feature_extractor, output_transformer)
        return (trainer, exporter)

    def _test_evaluator_ground_truth(
        self,
        dueling=False,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        environment = Gridworld()
        evaluator = GridworldEvaluator(environment, False, DISCOUNT)
        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
            environment, {}, dueling, use_gpu, use_all_avail_gpus, clip_grad_norm
        )
        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

    def test_evaluator_ground_truth_no_dueling_modular(self):
        self._test_evaluator_ground_truth()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_no_dueling_gpu_modular(self):
        self._test_evaluator_ground_truth(use_gpu=True)

    def test_evaluator_ground_truth_dueling_modular(self):
        self._test_evaluator_ground_truth(dueling=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_dueling_gpu_modular(self):
        self._test_evaluator_ground_truth(dueling=True, use_gpu=True)

    def _test_reward_boost(self, use_gpu=False, use_all_avail_gpus=False):
        environment = Gridworld()
        reward_boost = {"L": 100, "R": 200, "U": 300, "D": 400}
        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
            environment, reward_boost, False, use_gpu, use_all_avail_gpus
        )
        evaluator = GridworldEvaluator(
            env=environment, assume_optimal_policy=False, gamma=DISCOUNT
        )
        self.evaluate_gridworld(environment, evaluator, trainer, exporter, use_gpu)

    def test_reward_boost_modular(self):
        self._test_reward_boost()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reward_boost_gpu_modular(self):
        self._test_reward_boost(use_gpu=True)

    def _test_predictor_export(self):
        """Verify that q-values before model export equal q-values after
        model export. Meant to catch issues with export logic."""
        environment = Gridworld()
        samples = Samples(
            mdp_ids=["0"],
            sequence_numbers=[0],
            sequence_number_ordinals=[1],
            states=[{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 15: 1.0, 24: 1.0}],
            actions=["D"],
            action_probabilities=[0.5],
            rewards=[0],
            possible_actions=[["R", "D"]],
            next_states=[{5: 1.0}],
            next_actions=["U"],
            terminals=[False],
            possible_next_actions=[["R", "U", "D"]],
        )
        tdps = environment.preprocess_samples(samples, 1)

        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
            environment, {}, False
        )
        input = rlt.StateInput(state=rlt.FeatureVector(float_features=tdps[0].states))

        pre_export_q_values = trainer.q_network(input).q_values.detach().numpy()

        predictor = exporter.export()
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "model")
            predictor.save(tmp_path, "minidb")
            new_predictor = DQNPredictor.load(tmp_path, "minidb")

        post_export_q_values = new_predictor.predict([samples.states[0]])

        for i, action in enumerate(environment.ACTIONS):
            self.assertAlmostEqual(
                pre_export_q_values[0][i], post_export_q_values[0][action], places=4
            )

    def test_predictor_export_modular(self):
        self._test_predictor_export()
