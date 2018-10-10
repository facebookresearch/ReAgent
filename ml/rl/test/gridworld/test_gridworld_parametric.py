#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous
from ml.rl.test.gridworld.gridworld_continuous_enum import GridworldContinuousEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldContinuousEvaluator
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    FactorizationParameters,
    FeedForwardParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


class TestGridworldParametric(unittest.TestCase):
    def setUp(self):
        self.minibatch_size = 512
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

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

    def get_sarsa_trainer(
        self, environment, parameters=None, use_gpu=False, use_all_avail_gpus=False
    ):
        parameters = parameters or self.get_sarsa_parameters()
        return ParametricDQNTrainer(
            parameters,
            environment.normalization,
            environment.normalization_action,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )

    def _test_trainer_sarsa(self, use_gpu=False, use_all_avail_gpus=False):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment, use_gpu=use_gpu, use_all_avail_gpus=use_all_avail_gpus
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa(self):
        self._test_trainer_sarsa()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_gpu(self):
        self._test_trainer_sarsa(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_all_gpus(self):
        self._test_trainer_sarsa(use_gpu=True, use_all_avail_gpus=True)

    def _test_trainer_sarsa_factorized(self, use_gpu=False, use_all_avail_gpus=False):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment,
            self.get_sarsa_parameters_factorized(),
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_factorized(self):
        self._test_trainer_sarsa_factorized()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_factorized_gpu(self):
        self._test_trainer_sarsa_factorized(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_factorized_all_gpus(self):
        self._test_trainer_sarsa_factorized(use_gpu=True, use_all_avail_gpus=True)

    def _test_trainer_sarsa_enum(self, use_gpu=False, use_all_avail_gpus=False):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment, use_gpu=use_gpu, use_all_avail_gpus=use_all_avail_gpus
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum(self):
        self._test_trainer_sarsa_enum()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_enum_gpu(self):
        self._test_trainer_sarsa_enum(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_enum_all_gpus(self):
        self._test_trainer_sarsa_enum(use_gpu=True, use_all_avail_gpus=True)

    def _test_trainer_sarsa_enum_factorized(
        self, use_gpu=False, use_all_avail_gpus=False
    ):
        environment = GridworldContinuousEnum()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        trainer = self.get_sarsa_trainer(
            environment,
            self.get_sarsa_parameters_factorized(),
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        predictor = trainer.predictor()
        evaluator = GridworldContinuousEvaluator(
            environment, False, DISCOUNT, False, samples
        )
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp)

        predictor = trainer.predictor()
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.15)

    def test_trainer_sarsa_enum_factorized(self):
        self._test_trainer_sarsa_enum_factorized()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_enum_factorized_gpu(self):
        self._test_trainer_sarsa_enum_factorized(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trainer_sarsa_enum_factorized_all_gpus(self):
        self._test_trainer_sarsa_enum_factorized(use_gpu=True, use_all_avail_gpus=True)

    def _test_evaluator_ground_truth(self, use_gpu=False, use_all_avail_gpus=False):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        trainer = self.get_sarsa_trainer(
            environment, use_gpu=use_gpu, use_all_avail_gpus=use_all_avail_gpus
        )
        evaluator = Evaluator(None, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)

    def test_evaluator_ground_truth(self):
        self._test_evaluator_ground_truth()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_gpu(self):
        self._test_evaluator_ground_truth(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_all_gpus(self):
        self._test_evaluator_ground_truth(use_gpu=True, use_all_avail_gpus=True)

    def _test_evaluator_ground_truth_factorized(
        self, use_gpu=False, use_all_avail_gpus=False
    ):
        environment = GridworldContinuous()
        samples = environment.generate_samples(100000, 1.0, DISCOUNT)
        true_values = environment.true_values_for_sample(
            samples.states, samples.actions, False
        )
        # Hijack the reward timeline to insert the ground truth
        samples.episode_values = true_values
        trainer = self.get_sarsa_trainer(
            environment,
            self.get_sarsa_parameters_factorized(),
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        evaluator = Evaluator(None, 10, DISCOUNT, None, None)
        tdps = environment.preprocess_samples(
            samples, self.minibatch_size, use_gpu=use_gpu
        )

        for tdp in tdps:
            trainer.train(tdp, evaluator)

        self.assertLess(evaluator.mc_loss[-1], 0.15)

    def test_evaluator_ground_truth_factorized(self):
        self._test_evaluator_ground_truth_factorized()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_factorized_gpu(self):
        self._test_evaluator_ground_truth_factorized(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_evaluator_ground_truth_factorized_all_gpus(self):
        self._test_evaluator_ground_truth_factorized(
            use_gpu=True, use_all_avail_gpus=True
        )
