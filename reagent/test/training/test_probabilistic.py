#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import unittest

import numpy as np
import pytorch_lightning as pl
import reagent.core.types as rlt
import torch
import torch.optim as optim
from reagent.models.probabilistic_fully_connected_network import (
    FullyConnectedProbabilisticNetwork,
)
from reagent.training.cfeval.bayes_by_backprop_trainer import BayesByBackpropTrainer

logger = logging.getLogger(__name__)


def toy_function(x):
    return -(x**4) + 3 * x**2 - 5 * np.sin(x) + 1


class TestBayesByBackpropTraining(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(123)

    def test_probabilistic_network(self):
        net = FullyConnectedProbabilisticNetwork(
            [2, 16, 16, 1], ["relu", "relu", "linear"], prior_var=1
        )
        net = net
        trainer = BayesByBackpropTrainer(net)

        epochs = 1000
        optimizer = optim.Adam(net.parameters(), lr=0.1)
        batch_size = 6
        action = torch.ones(batch_size, 1)
        loss_ema = -1
        prev_loss_ema = -1

        for epoch in range(epochs):  # loop over the dataset multiple times
            x = torch.rand((batch_size, 1)) * 4 - 2
            y = toy_function(x)
            batch = rlt.BanditRewardModelInput(
                action=action,
                reward=y,
                state=rlt.FeatureData(float_features=x),
                # BanditRewardModelInput for simple fully supervised regression task
            )
            loss = next(trainer.train_step_gen(batch, epoch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss_ema == -1:
                loss_ema = loss
            else:
                loss_ema = loss_ema * 0.99 + 0.01 * loss
            if epoch % 100 == 1:
                print("Loss EMA", loss_ema)
                if epoch > 250:  # give some time for training to stabilize
                    assert loss_ema < prev_loss_ema
                prev_loss_ema = loss_ema
        print("Finished Training")

        # test model confidence
        num_samples_per_point = 100
        test_domain = 25
        num_points = 100
        x_tmp = torch.cat(
            [
                torch.linspace(-1 * test_domain, test_domain, num_points).reshape(
                    -1, 1
                ),
                torch.ones((num_points, 1)),
            ],
            1,
        )
        y_samp = np.zeros((num_samples_per_point, num_points))
        for s in range(num_samples_per_point):
            y_tmp = net(x_tmp).cpu().detach().numpy()
            y_samp[s] = y_tmp.reshape(-1)
        mean = np.mean(y_samp, 0, keepdims=True)
        var = np.mean((y_samp - mean) ** 2, 0)

        # make several assertions that the further you get from the training domain, the more
        # unconfident the network becomes, eventually becoming extremely unconfident when x=-25
        assert var[60] > var[50]
        assert var[0] > var[60]
        assert var[0] > var[50] * 10
