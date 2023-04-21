#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import torch
from reagent.models.cb_base_model import UCBBaseModel
from reagent.models.fully_connected_network import FullyConnectedNetwork


logger = logging.getLogger(__name__)


class CBFullyConnectedNetwork(UCBBaseModel):
    """
    A Fully Connected Network (FCNN) for contextual bandits.
    It outputs a dictionary consisting of {pred_reward, pred_sigma, ucb}.
    This model does not yields pred_sigma as LinUCB or DeepRepresentLinearRegressionUCB does.
    It outputs a dummy or zero pred_sigma (uncertainty=ucb_alpha*pred_sigma) by setting pred_sigma=0.
    As a result, this model outputs ucb=pred_reward.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        *,
        activation: str = "relu",
        output_activation: str = "linear",
        use_batch_norm: bool = False,
        min_std: float = 0.0,
        dropout_ratio: float = 0.0,
        use_layer_norm: bool = False,
        normalize_output: bool = False,
        orthogonal_init: bool = False,
        use_skip_connections: bool = True,
    ):
        super().__init__(input_dim=input_dim)
        self.net = FullyConnectedNetwork(
            layers=[input_dim] + hidden_layer_dims + [1],  # size(output layer)=1
            activations=[activation] * len(hidden_layer_dims) + [output_activation],
            use_batch_norm=use_batch_norm,
            min_std=min_std,
            dropout_ratio=dropout_ratio,
            use_layer_norm=use_layer_norm,
            normalize_output=normalize_output,
            orthogonal_init=orthogonal_init,
            use_skip_connections=use_skip_connections,
        )

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = 0.0
    ) -> Dict[str, torch.Tensor]:
        if ucb_alpha != 0:
            logger.warn(
                f"CBFullyConnectedNetwork supports only point predictions (ucb_alpha=0), but ucb_alpha={ucb_alpha} was used"
            )
        pred_reward = self.net(inp).squeeze(-1)
        pred_sigma = torch.zeros_like(pred_reward)
        ucb = pred_reward
        return {"pred_reward": pred_reward, "pred_sigma": pred_sigma, "ucb": ucb}
