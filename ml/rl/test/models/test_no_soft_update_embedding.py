#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import unittest

import numpy.testing as npt
import torch
import torch.nn as nn
from ml.rl.models.no_soft_update_embedding import NoSoftUpdateEmbedding
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters, RLParameters
from ml.rl.training.rl_trainer_pytorch import RLTrainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = NoSoftUpdateEmbedding(10, 3)

    def forward(self, input):
        return self.embedding(input)


class TestNoSoftUpdteEmbedding(unittest.TestCase):
    def test_no_soft_update(self):
        model = Model()
        target_model = copy.deepcopy(model)

        for target_param, param in zip(model.parameters(), target_model.parameters()):
            self.assertIs(target_param, param)

        optimizer = torch.optim.Adam(model.parameters())

        x = torch.tensor([1, 2], dtype=torch.int64)
        emb = model(x)

        loss = emb.sum()

        loss.backward()
        optimizer.step()

        params = list(model.parameters())
        self.assertEqual(1, len(params))
        param = params[0].detach().numpy()

        trainer = RLTrainer(
            DiscreteActionModelParameters(rl=RLParameters()), use_gpu=False
        )
        trainer._soft_update(model, target_model, 0.1)

        target_params = list(target_model.parameters())
        self.assertEqual(1, len(target_params))
        target_param = target_params[0].detach().numpy()

        npt.assert_array_equal(target_param, param)
