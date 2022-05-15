#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from reagent.core import types as rlt
from reagent.models.fully_connected_network import FloatFeatureFullyConnected


INVALID_ACTION_CONSTANT: float = -1e10


class FullyConnectedDQN(FloatFeatureFullyConnected):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        *,
        output_activation: str = "linear",
        num_atoms: Optional[int] = None,
        use_batch_norm: bool = False,
        dropout_ratio: float = 0.0,
        normalized_output: bool = False,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            output_dim=action_dim,
            sizes=sizes,
            activations=activations,
            num_atoms=num_atoms,
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
            normalized_output=normalized_output,
            use_layer_norm=use_layer_norm,
            output_activation=output_activation,
        )
        self.action_dim = self.output_dim

    def forward(
        self,
        state: rlt.FeatureData,
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = super().forward(state=state)
        if possible_actions_mask is not None:
            # subtract huge value from impossible actions to force their probabilities to 0
            x = x + (1 - possible_actions_mask.float()) * INVALID_ACTION_CONSTANT
        return x
