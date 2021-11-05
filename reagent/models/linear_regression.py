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
        predict_ucb: If True, the model outputs an Upper Confidence Bound (UCB).
            If False, the model outputs the point estimate
        ucb_alpha: The coefficient on the standard deviation in UCB formula.
            Only used if predict_ucb=True.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        l2_reg_lambda: float = 1.0,
        predict_ucb: float = False,
        ucb_alpha: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.predict_ucb = predict_ucb
        self.ucb_alpha = ucb_alpha
        self.A = l2_reg_lambda * torch.eye(self.input_dim)
        self.b = torch.zeros(self.input_dim)
        self.coefs = torch.zeros(self.input_dim)
        self.inv_A = torch.zeros(self.input_dim, self.input_dim)
        self.coefs_valid_for_A = -torch.ones_like(
            self.A
        )  # value of A matrix for which self.coefs were estimated

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim)

    def _estimate_coefs(self):
        """
        Compute current estimate of regression coefficients and A_inv=A**-1
        We save both coefficients and A_inv in case they are needed again before we add observations
        """
        self.inv_A = torch.inverse(self.A)
        self.coefs = torch.matmul(self.inv_A, self.b)
        self.coefs_valid_for_A = self.A.clone()

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward can return the mean or a UCB. If returning UCB, the CI width is stddev*ucb_alpha
        If ucb_alpha is not passed in, a fixed alpha from init is used
        """
        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha
        if not (self.coefs_valid_for_A == self.A).all():
            self._estimate_coefs()
        if self.predict_ucb:
            return torch.matmul(inp, self.coefs) + ucb_alpha * torch.sqrt(
                batch_quadratic_form(inp, self.inv_A)
            )
        else:
            return torch.matmul(inp, self.coefs)
