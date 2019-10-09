#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
import tempfile
import unittest

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.models.categorical_dqn import CategoricalDQN
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.models.dueling_quantile_dqn import DuelingQuantileDQN
from ml.rl.models.output_transformer import DiscreteActionOutputTransformer
from ml.rl.models.quantile_dqn import QuantileDQN
from ml.rl.parameters import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.prediction.dqn_torch_predictor import DiscreteDqnTorchPredictor
from ml.rl.prediction.predictor_wrapper import (
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
)
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import DISCOUNT, Samples
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.test.gridworld.gridworld_test_base import GridworldTestBase
from ml.rl.torch_utils import export_module_to_buffer
from ml.rl.training.c51_trainer import C51Trainer
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.qrdqn_trainer import QRDQNTrainer
from ml.rl.training.rl_exporter import DQNExporter


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

        if quantile:
            if dueling:
                q_network = DuelingQuantileDQN(
                    layers=[get_num_output_features(environment.normalization)]
                    + parameters.training.layers[1:-1]
                    + [len(environment.ACTIONS)],
                    activations=parameters.training.activations,
                    num_atoms=parameters.rainbow.num_atoms,
                )
            else:
                q_network = QuantileDQN(
                    state_dim=get_num_output_features(environment.normalization),
                    action_dim=len(environment.ACTIONS),
                    num_atoms=parameters.rainbow.num_atoms,
                    sizes=parameters.training.layers[1:-1],
                    activations=parameters.training.activations[:-1],
                )
        elif categorical:
            assert not dueling
            q_network = CategoricalDQN(
                state_dim=get_num_output_features(environment.normalization),
                action_dim=len(environment.ACTIONS),
                num_atoms=parameters.rainbow.num_atoms,
                qmin=-100,
                qmax=200,
                sizes=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
            )
        else:
            if dueling:
                q_network = DuelingQNetwork(
                    layers=[get_num_output_features(environment.normalization)]
                    + parameters.training.layers[1:-1]
                    + [len(environment.ACTIONS)],
                    activations=parameters.training.activations,
                )
            else:
                q_network = FullyConnectedDQN(
                    state_dim=get_num_output_features(environment.normalization),
                    action_dim=len(environment.ACTIONS),
                    sizes=parameters.training.layers[1:-1],
                    activations=parameters.training.activations[:-1],
                )

        q_network_cpe, q_network_cpe_target, reward_network = None, None, None

        if parameters.evaluation and parameters.evaluation.calc_cpe_in_training:
            q_network_cpe = FullyConnectedDQN(
                state_dim=get_num_output_features(environment.normalization),
                action_dim=len(environment.ACTIONS),
                sizes=parameters.training.layers[1:-1],
                activations=parameters.training.activations[:-1],
            )
            q_network_cpe_target = q_network_cpe.get_target_network()
            reward_network = FullyConnectedDQN(
                state_dim=get_num_output_features(environment.normalization),
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
            trainer = C51Trainer(
                q_network, q_network.get_target_network(), parameters, use_gpu
            )
        else:
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

    def get_modular_sarsa_trainer_exporter(
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
        parameters = self.get_sarsa_parameters(
            environment, reward_shape, dueling, categorical, quantile, clip_grad_norm
        )
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
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=environment.normalization
        )
        output_transformer = DiscreteActionOutputTransformer(parameters.actions)
        exporter = DQNExporter(trainer.q_network, feature_extractor, output_transformer)
        return (trainer, exporter)

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
        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
            environment,
            {},
            dueling=dueling,
            categorical=categorical,
            quantile=quantile,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
            clip_grad_norm=clip_grad_norm,
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
        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
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
            environment, {}, False, False, False
        )
        input = rlt.PreprocessedState.from_tensor(tdps[0].states)

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

        trainer, exporter = self.get_modular_sarsa_trainer_exporter(
            environment, {}, False, False, False
        )
        input = rlt.PreprocessedState.from_tensor(tdps[0].states)

        pre_export_q_values = trainer.q_network(input).q_values.detach().numpy()

        preprocessor = Preprocessor(environment.normalization, False)
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
