#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import torch
from reagent.models.base import ModelBase


logger = logging.getLogger(__name__)


def batch_quadratic_form(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute the quadratic form x^T * A * x for a batched input x.
    Inspired by https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
    This is a vectorized implementation of out[i] = x[i].t() @ A @ x[i]
    x shape: (B, N)
    A shape: (N, N)
    output shape: (B)
    """
    return (torch.matmul(x, A) * x).sum(1)


class LinearRegressionUCB(ModelBase):
    """
    A linear regression model for LinUCB.
    Note that instead of being trained by a PyTorch optimizer, we explicitly
        update attributes A and b (according to the LinUCB formulas implemented in
        reagent.training.cb.linucb_trainer.LinUCBTrainer).
    Since computing the regression coefficients inverse matrix inversion (expensive op), we
        save time by only computing the coefficients when necessary (when doing inference).

    Args:
        input_dim: Dimension of input data
        l2_reg_lambda: The weight on L2 regularization
        ucb_alpha: The coefficient on the standard deviation in UCB formula.
            Set it to 0 to predict the expected value instead of UCB.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        l2_reg_lambda: float = 1.0,
        ucb_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.ucb_alpha = ucb_alpha
        self.register_buffer("A", l2_reg_lambda * torch.eye(self.input_dim))
        self.register_buffer("b", torch.zeros(self.input_dim))
        self.register_buffer("_coefs", torch.zeros(self.input_dim))
        self.register_buffer("inv_A", torch.zeros(self.input_dim, self.input_dim))
        self.register_buffer(
            "coefs_valid_for_A", -torch.ones((self.input_dim, self.input_dim))
        )  # value of A matrix for which self.coefs were estimated

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim)

    def _calculate_coefs(self) -> None:
        """
        Compute current estimate of regression coefficients and A_inv=A**-1
        We save both coefficients and A_inv in case they are needed again before we add observations
        The coefficients are computed only when needed because their computation can be expensive
            (involves matrix inversion)
        """
        self.inv_A = torch.inverse(self.A)
        self._coefs = torch.matmul(self.inv_A, self.b)
        self.coefs_valid_for_A = self.A.clone()

    @property
    def coefs(self) -> torch.Tensor:
        if not (self.coefs_valid_for_A == self.A).all():
            # re-calculate the coefs only if the previous value is invalid
            self._calculate_coefs()
        return self._coefs

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward can return the mean or a UCB. If returning UCB, the CI width is stddev*ucb_alpha
        If ucb_alpha is not passed in, a fixed alpha from init is used
        """
        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha

        mu = torch.matmul(inp, self.coefs)

        if ucb_alpha != 0:
            return mu + ucb_alpha * torch.sqrt(batch_quadratic_form(inp, self.inv_A))
        else:
            return mu
