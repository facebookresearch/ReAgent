#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

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
    ) -> Dict[str, torch.Tensor]:
        """
        Model forward pass. For UCB models should return the prediction with uncertainty, not the point prediction.
        Returns pred_label, pred_sigma, ucb (where ucb = pred_label + ucb_alpha*pred_sigma)
        """
        pass

    def forward_inference(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        This forward method will be called by the inference wrapper.
        By default it's same as regular forward(), but users can override it
            if they need special behavior in the inference wrapper.
        """
        return self.forward(inp, ucb_alpha=ucb_alpha)
