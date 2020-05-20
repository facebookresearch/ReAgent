#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import numpy as np
import reagent.models as models
import reagent.types as rlt
import torch
from reagent.parameters import (
    DiscreteActionModelParameters,
    NormalizationData,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from reagent.prediction.dqn_torch_predictor import DiscreteDqnTorchPredictor
from reagent.prediction.predictor_wrapper import (
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
)
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.test.gridworld.gridworld import Gridworld
from reagent.test.gridworld.gridworld_base import DISCOUNT, Samples
from reagent.test.gridworld.gridworld_evaluator import GridworldEvaluator
from reagent.test.gridworld.gridworld_test_base import GridworldTestBase
from reagent.torch_utils import export_module_to_buffer
from reagent.training.c51_trainer import C51Trainer, C51TrainerParameters
from reagent.training.dqn_trainer import DQNTrainer, DQNTrainerParameters
from reagent.training.qrdqn_trainer import QRDQNTrainer, QRDQNTrainerParameters


class TestGridworld(GridworldTestBase):
    def setUp(self):
        self.minibatch_size = 512
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        super().setUp()

    def get_sarsa_parameters(
        self, environment, reward_shape, dueling, categorical, quantile, clip_grad_norm
    ):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=1.0,
            maxq_learning=False,
            reward_boost=reward_shape,
        )
        training_parameters = TrainingParameters(
            layers=[-1, 128, -1] if dueling else [-1, -1],
            activations=["relu", "relu"] if dueling else ["linear"],
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
                double_q_learning=True,
                dueling_architecture=dueling,
                categorical=categorical,
                quantile=quantile,
                num_atoms=5,
            ),
        )

    def get_modular_sarsa_trainer(
        self,
        environment,
        dueling,
        categorical,
        quantile,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        return self.get_modular_sarsa_trainer_reward_boost(
            environment,
            {},
            dueling=dueling,
            categorical=categorical,
            quantile=quantile,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
        )

    def get_modular_sarsa_trainer_reward_boost(
        self,
        environment,
        reward_shape,
        dueling,
        categorical,
        quantile,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        assert not quantile or not categorical
        parameters = self.get_sarsa_parameters(
            environment, reward_shape, dueling, categorical, quantile, clip_grad_norm
        )

        state_normalization_parameters = environment.normalization

        def make_dueling_dqn(num_atoms=None):
            return models.DuelingQNetwork.make_fully_connected(
                state_dim=get_num_output_features(state_normalization_parameters),
                action_dim=len(environment.ACTIONS),
                layers=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
                num_atoms=num_atoms,
            )

        if quantile:
            if dueling:
                q_network = make_dueling_dqn(num_atoms=parameters.rainbow.num_atoms)

            else:
                q_network = models.FullyConnectedDQN(
                    state_dim=get_num_output_features(state_normalization_parameters),
                    action_dim=len(environment.ACTIONS),
                    num_atoms=parameters.rainbow.num_atoms,
                    sizes=parameters.training.layers[1:-1],
                    activations=parameters.training.activations[:-1],
                )
        elif categorical:
            assert not dueling
            distributional_network = models.FullyConnectedDQN(
                state_dim=get_num_output_features(state_normalization_parameters),
                action_dim=len(environment.ACTIONS),
                num_atoms=parameters.rainbow.num_atoms,
                sizes=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
            )
            q_network = models.CategoricalDQN(
                distributional_network,
                qmin=-100,
                qmax=200,
                num_atoms=parameters.rainbow.num_atoms,
            )
        else:
            if dueling:
                q_network = make_dueling_dqn()
            else:
                q_network = models.FullyConnectedDQN(
                    state_dim=get_num_output_features(state_normalization_parameters),
                    action_dim=len(environment.ACTIONS),
                    sizes=parameters.training.layers[1:-1],
                    activations=parameters.training.activations[:-1],
                )

        q_network_cpe, q_network_cpe_target, reward_network = None, None, None

        if parameters.evaluation and parameters.evaluation.calc_cpe_in_training:
            q_network_cpe = models.FullyConnectedDQN(
                state_dim=get_num_output_features(state_normalization_parameters),
                action_dim=len(environment.ACTIONS),
                sizes=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
            )
            q_network_cpe_target = q_network_cpe.get_target_network()
            reward_network = models.FullyConnectedDQN(
                state_dim=get_num_output_features(state_normalization_parameters),
                action_dim=len(environment.ACTIONS),
                sizes=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
            )

        if use_gpu:
            q_network = q_network.cuda()
            if parameters.evaluation.calc_cpe_in_training:
                reward_network = reward_network.cuda()
                q_network_cpe = q_network_cpe.cuda()
                q_network_cpe_target = q_network_cpe_target.cuda()
            if use_all_avail_gpus and not categorical:
                q_network = q_network.get_distributed_data_parallel_model()
                reward_network = reward_network.get_distributed_data_parallel_model()
                q_network_cpe = q_network_cpe.get_distributed_data_parallel_model()
                q_network_cpe_target = (
                    q_network_cpe_target.get_distributed_data_parallel_model()
                )

        if quantile:
            parameters = QRDQNTrainerParameters.from_discrete_action_model_parameters(
                parameters
            )
            trainer = QRDQNTrainer(
                q_network,
                q_network.get_target_network(),
                parameters,
                use_gpu,
                reward_network=reward_network,
                q_network_cpe=q_network_cpe,
                q_network_cpe_target=q_network_cpe_target,
            )
        elif categorical:
            parameters = C51TrainerParameters.from_discrete_action_model_parameters(
                parameters
            )
            trainer = C51Trainer(
                q_network, q_network.get_target_network(), parameters, use_gpu
            )
        else:
            parameters = DQNTrainerParameters.from_discrete_action_model_parameters(
                parameters
            )
            trainer = DQNTrainer(
                q_network,
                q_network.get_target_network(),
                reward_network,
                parameters,
                use_gpu,
                q_network_cpe=q_network_cpe,
                q_network_cpe_target=q_network_cpe_target,
            )
        return trainer

    def get_trainer(
        self,
        environment,
        reward_shape,
        dueling,
        categorical,
        quantile,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        trainer = self.get_modular_sarsa_trainer_reward_boost(
            environment,
            reward_shape,
            dueling=dueling,
            categorical=categorical,
            quantile=quantile,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
        )
        return trainer

    def get_predictor(self, trainer, environment):
        state_normalization_parameters = environment.normalization
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        q_network = trainer.q_network
        if isinstance(trainer, QRDQNTrainer):

            class _Mean(torch.nn.Module):
                def forward(self, input):
                    assert input.ndim == 3
                    return input.mean(dim=2)

            q_network = models.Sequential(q_network, _Mean())

        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor
        )
        serving_module = DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor,
            action_names=environment.ACTIONS,
        )
        predictor = DiscreteDqnTorchPredictor(serving_module)
        return predictor

    def _test_evaluator_ground_truth(
        self,
        dueling=False,
        categorical=False,
        quantile=False,
        use_gpu=False,
        use_all_avail_gpus=False,
        clip_grad_norm=None,
    ):
        environment = Gridworld()
        evaluator = GridworldEvaluator(environment, False, DISCOUNT)
        trainer = self.get_trainer(
            environment,
            {},
            dueling=dueling,
            categorical=categorical,
            quantile=quantile,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
        )
        self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

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

    def test_evaluator_ground_truth_categorical_modular(self):
        self._test_evaluator_ground_truth(categorical=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_categorical_gpu_modular(self):
        self._test_evaluator_ground_truth(categorical=True, use_gpu=True)

    def test_evaluator_ground_truth_quantile_modular(self):
        self._test_evaluator_ground_truth(quantile=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_quantile_gpu_modular(self):
        self._test_evaluator_ground_truth(quantile=True, use_gpu=True)

    def test_evaluator_ground_truth_dueling_quantile_modular(self):
        self._test_evaluator_ground_truth(dueling=True, quantile=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_dueling_quantile_gpu_modular(self):
        self._test_evaluator_ground_truth(dueling=True, quantile=True, use_gpu=True)

    def _test_reward_boost(self, use_gpu=False, use_all_avail_gpus=False):
        environment = Gridworld()
        reward_boost = {"L": 100, "R": 200, "U": 300, "D": 400}
        trainer = self.get_trainer(
            environment,
            reward_boost,
            dueling=False,
            categorical=False,
            quantile=False,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        evaluator = GridworldEvaluator(
            env=environment, assume_optimal_policy=False, gamma=DISCOUNT
        )
        self.evaluate_gridworld(environment, evaluator, trainer, use_gpu)

    def test_reward_boost_modular(self):
        self._test_reward_boost()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reward_boost_gpu_modular(self):
        self._test_reward_boost(use_gpu=True)

    def test_predictor_torch_export(self):
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
        assert len(tdps) == 1, "Invalid number of data pages"

        trainer = self.get_trainer(environment, {}, False, False, False)
        input = rlt.FeatureData(tdps[0].states)

        pre_export_q_values = trainer.q_network(input).detach().numpy()

        state_normalization_parameters = environment.normalization
        preprocessor = Preprocessor(state_normalization_parameters, False)
        cpu_q_network = trainer.q_network.cpu_model()
        cpu_q_network.eval()
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(cpu_q_network, preprocessor)
        serving_module = DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names=environment.ACTIONS
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            buf = export_module_to_buffer(serving_module)
            tmp_path = os.path.join(tmpdirname, "model")
            with open(tmp_path, "wb") as f:
                f.write(buf.getvalue())
                f.close()
                predictor = DiscreteDqnTorchPredictor(torch.jit.load(tmp_path))

        post_export_q_values = predictor.predict([samples.states[0]])

        for i, action in enumerate(environment.ACTIONS):
            self.assertAlmostEqual(
                float(pre_export_q_values[0][i]),
                float(post_export_q_values[0][action]),
                places=4,
            )
