#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest
from unittest.mock import MagicMock

import numpy as np

import numpy.testing as npt

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from reagent.core.types import CBInput
from reagent.evaluation.cb.policy_evaluator import PolicyEvaluator
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.linear_regression import LinearRegressionUCB
from reagent.training.cb.linucb_trainer import LinUCBTrainer


class TestEvalDuringTraining(unittest.TestCase):
    def setUp(self):
        x_dim = 2
        self.policy_network = LinearRegressionUCB(x_dim)
        policy = Policy(scorer=self.policy_network, sampler=GreedyActionSampler())

        self.trainer = LinUCBTrainer(policy)
        self.eval_module = PolicyEvaluator(self.policy_network)
        logger = TensorBoardLogger("/tmp/tb")
        logger.log_metrics = MagicMock()
        self.eval_module.attach_logger(logger)
        self.trainer.attach_eval_module(self.eval_module)

    def test_eval_during_training(self):
        """
        Test integration of evaluation into the training loop.

        run simulated training-evaluation sequence with 3 batches.
        the model and features are set up so that the model always selects action 1, so all observations
            with logged action 0 will be skipped
        the evaluated model is updated only after consuming batch_1
        """

        # step 1: consume batch_1. just the 2nd data point will be used
        batch_1 = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2],
                        [1, 3],
                    ],
                    [
                        [1, 4],
                        [1, 5],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
        )
        self.trainer.training_step(batch_1, 0)

        # flush the buffers inside the model
        self.trainer.scorer._calculate_coefs()
        self.eval_module.eval_model._calculate_coefs()

        # update the evaluated model. this model will be used for evaluation (action selection) in both batch_2 and batch_3 bcs we won't update it after batch_2
        self.eval_module.update_eval_model(self.trainer.scorer)

        """
        after batch_1, the model has state:
        A: tensor([[ 1.,  5.],
                [ 5., 25.]])
        inv_A: tensor([[ 0.9630, -0.1852],
                [-0.1852,  0.0741]])
        b: tensor([ 2.3000, 11.5000])
        num_obs: tensor([1.])
        coefs: tensor([0.0852, 0.4259])
        """

        # check if trained model state is correct
        batch_1_used_features = batch_1.context_arm_features[1, 1].numpy()
        npt.assert_allclose(
            (self.trainer.scorer.avg_A * self.trainer.scorer.sum_weight).numpy(),
            np.outer(batch_1_used_features, batch_1_used_features)
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (self.trainer.scorer.avg_b * self.trainer.scorer.sum_weight).numpy(),
            batch_1_used_features
            * batch_1.reward[1, 0].item()
            * 2,  # *2 due to importance weight applied during offline eval
        )

        # check if evaluated model state is correct (should be same as trained model)
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_A
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            np.outer(batch_1_used_features, batch_1_used_features)
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_b
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            batch_1_used_features
            * batch_1.reward[1, 0].item()
            * 2,  # *2 due to importance weight applied during offline eval
        )

        # step 2: consume batch_2. just the 1st data point will be used
        batch_2 = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 7],
                        [1, 8],
                        [
                            1,
                            9,
                        ],  # this arm would have been chosen by the model if it was present
                    ],
                    [
                        [1, 9],
                        [1, 10],
                        [1, 11],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[1], [0]], dtype=torch.long),
            reward=torch.tensor([[1.2], [2.9]], dtype=torch.float),
            arm_presence=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        )
        self.trainer.training_step(batch_2, 0)

        # flush the buffers inside the model
        self.trainer.scorer._calculate_coefs()
        self.eval_module.eval_model._calculate_coefs()

        # check that trained model state is correct
        batch_2_used_features = batch_2.context_arm_features[0, 1].numpy()
        npt.assert_allclose(
            (self.trainer.scorer.avg_A * self.trainer.scorer.sum_weight).numpy(),
            (
                np.outer(batch_1_used_features, batch_1_used_features)
                + np.outer(batch_2_used_features, batch_2_used_features)
            )
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (self.trainer.scorer.avg_b * self.trainer.scorer.sum_weight).numpy(),
            (
                batch_1_used_features * batch_1.reward[1, 0].item()
                + batch_2_used_features * batch_2.reward[0, 0].item()
            )
            * 2,  # *2 due to importance weight applied during offline eval
        )

        # check that evaluated model state is correct (same as it was after batch_1)
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_A
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            np.outer(batch_1_used_features, batch_1_used_features)
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_b
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            batch_1_used_features
            * batch_1.reward[1, 0].item()
            * 2,  # *2 due to importance weight applied during offline eval
        )

        """
        after batch_2, the model has state:
        A: tensor([[ 2., 13.],
                [13., 89.]])
        inv_A: tensor([[ 0.8911, -0.1287],
                [-0.1287,  0.0297]])
        b: tensor([ 3.5000, 21.1000])
        num_obs: tensor([2.])
        coefs: tensor([0.4030, 0.1762])
        """

        # step 3: consume batch_3. just the 2nd data point will be used
        batch_3 = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 7],
                        [1, 8],
                    ],
                    [
                        [1, 9],
                        [1, 10],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[3.5], [7.1]], dtype=torch.float),
        )
        self.trainer.training_step(batch_3, 0)

        # flush the buffers inside the model
        self.trainer.scorer._calculate_coefs()
        self.eval_module.eval_model._calculate_coefs()

        # finish training epoch
        self.trainer.on_train_epoch_end()

        """
        after batch_3, the model has state:
        A: tensor([[  3.,  23.],
                [ 23., 189.]])
        inv_A: tensor([[ 0.8225, -0.0996],
                [-0.0996,  0.0173]])
        b: tensor([10.8000, 94.1000])
        num_obs: tensor([3.])
        coefs: tensor([-0.4861,  0.5541])
        """

        # check that trained model state is correct
        batch_3_used_features = batch_3.context_arm_features[1, 1].numpy()
        npt.assert_allclose(
            (self.trainer.scorer.avg_A * self.trainer.scorer.sum_weight).numpy(),
            (
                np.outer(batch_1_used_features, batch_1_used_features)
                + np.outer(batch_2_used_features, batch_2_used_features)
                + np.outer(batch_3_used_features, batch_3_used_features)
            )
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (self.trainer.scorer.avg_b * self.trainer.scorer.sum_weight).numpy(),
            (
                batch_1_used_features * batch_1.reward[1, 0].item()
                + batch_2_used_features * batch_2.reward[0, 0].item()
                + batch_3_used_features * batch_3.reward[1, 0].item()
            )
            * 2,  # *2 due to importance weight applied during offline eval
        )

        # check that evaluated model state is correct (same as it was after batch_1)
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_A
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            np.outer(batch_1_used_features, batch_1_used_features)
            * 2,  # *2 due to importance weight applied during offline eval
        )
        npt.assert_allclose(
            (
                self.eval_module.eval_model.avg_b
                * self.eval_module.eval_model.sum_weight
            ).numpy(),
            batch_1_used_features
            * batch_1.reward[1, 0].item()
            * 2,  # *2 due to importance weight applied during offline eval
        )

        # check average reward. should be AVG([2.3, 1.2, 7.1]) = 4.2
        self.assertAlmostEqual(
            self.eval_module.get_avg_reward(),
            np.mean(
                [
                    batch_1.reward[1, 0].item(),
                    batch_2.reward[0, 0].item(),
                    batch_3.reward[1, 0].item(),
                ]
            ),
            places=4,
        )

        # check total weight (number of observations). Should be 3
        self.assertAlmostEqual(
            (
                self.eval_module.sum_weight_accepted
                + self.eval_module.sum_weight_accepted_local
            ).item(),
            3.0,
            places=4,
        )

        # metrics should have been logged once, at the end of epoch
        # TODO: test logging logic triggered by eval_model_update_critical_weight
        self.eval_module.logger.log_metrics.assert_called_once()
