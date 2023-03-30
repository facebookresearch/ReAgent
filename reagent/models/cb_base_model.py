#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class UCBBaseModel(torch.nn.Module, ABC):
    """
    Abstract base class for UCB-style CB models.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim)

    @abstractmethod
    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Model forward pass. For UCB models should return the UCB, not the point prediction.
        """
        pass

    def get_point_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Get point prediction from the model (ignoring uncertainty)
        """
        return self(inp, ucb_alpha=0.0)

    def forward_inference(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        This forward method will be called by the inference wrapper.
        By default it's same as regular forward(), but users can override it
            if they need special behavior in the inference wrapper.
        """
        return self.forward(inp, ucb_alpha=ucb_alpha)
