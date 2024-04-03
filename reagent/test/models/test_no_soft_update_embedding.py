#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import copy
import unittest

import numpy.testing as npt
import torch
import torch.nn as nn
from reagent.models.no_soft_update_embedding import NoSoftUpdateEmbedding


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = NoSoftUpdateEmbedding(10, 3)

    def forward(self, input):
        return self.embedding(input)


class TestNoSoftUpdteEmbedding(unittest.TestCase):
    def test_no_soft_update(self) -> None:
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

        self._soft_update(model, target_model, 0.1)

        target_params = list(target_model.parameters())
        self.assertEqual(1, len(target_params))
        target_param = target_params[0].detach().numpy()

        npt.assert_array_equal(target_param, param)

    @torch.no_grad()
    def _soft_update(self, network, target_network, tau) -> None:
        """Target network update logic as defined in DDPG paper
        updated_params = tau * network_params + (1 - tau) * target_network_params
        :param network network with parameters to include in soft update
        :param target_network target network with params to soft update
        :param tau hyperparameter to control target tracking speed
        """
        for t_param, param in zip(target_network.parameters(), network.parameters()):
            if t_param is param:
                # Skip soft-updating when the target network shares the parameter with
                # the network being train.
                continue
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)
