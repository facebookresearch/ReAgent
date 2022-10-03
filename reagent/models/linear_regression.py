#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import torch
from pytorch_lightning.utilities.distributed import ReduceOp, sync_ddp_if_available
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
        gamma: per-epoch discount factor (A and b get multiplied by gamma every epoch)
    """

    def __init__(
        self,
        input_dim: int,
        *,
        l2_reg_lambda: float = 1.0,
        ucb_alpha: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.ucb_alpha = ucb_alpha
        self.l2_reg_lambda = l2_reg_lambda
        self.gamma = gamma
        assert self.gamma <= 1.0 and self.gamma > 0.0

        # the buffers below are split between "all data" and "current epoch" values. This is done
        #   to enable distributed training. "current epoch" values get summed acorss all trainers at
        #   the end of an epoch, but "all data" values don't need to be summed (they were already summed
        #   when the value got moved from "current epoch" to "all data")
        # A is sum of X^T*X across all data
        self.register_buffer("A", torch.zeros(self.input_dim, self.input_dim))
        # b is sum of reward*X across all data
        self.register_buffer("b", torch.zeros(self.input_dim))
        # A is sum of X^T*X across current epoch
        self.register_buffer("cur_A", torch.zeros(self.input_dim, self.input_dim))
        # b is sum of reward*X across curernt epoch
        self.register_buffer("cur_b", torch.zeros(self.input_dim))
        self.register_buffer("_coefs", torch.zeros(self.input_dim))
        self.register_buffer("inv_A", torch.zeros(self.input_dim, self.input_dim))
        self.register_buffer(
            "coefs_valid_for_A", -torch.ones((self.input_dim, self.input_dim))
        )  # value of A matrix for which self.coefs were estimated
        self.register_buffer("num_obs", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("cur_num_obs", torch.zeros(1, dtype=torch.int64))

        # add a dummy parameter so that DDP doesn't compain about lack of parameters with gradient
        self.dummy_param = torch.nn.parameter.Parameter(torch.zeros(1))

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim)

    def _calculate_coefs(self) -> None:
        """
        Compute current estimate of regression coefficients and A_inv=A**-1
        We save both coefficients and A_inv in case they are needed again before we add observations
        The coefficients are computed only when needed because their computation can be expensive
            (involves matrix inversion)
        """
        # reduce (sum) the values of `A` and `b` across all trainer processes.
        # Since `A` and `b` are computed as sums across all observations within each trainer,
        #   the correct reduction across trainers is the sum.
        # The coefficients can't be averaged across trainers because they are a non-linear
        #   function of `A` and `b`
        self.A += sync_ddp_if_available(self.cur_A, reduce_op=ReduceOp.SUM)
        self.b += sync_ddp_if_available(self.cur_b, reduce_op=ReduceOp.SUM)
        self.num_obs += sync_ddp_if_available(self.cur_num_obs, reduce_op=ReduceOp.SUM)

        self.inv_A = torch.inverse(
            self.A
            + self.l2_reg_lambda
            * torch.eye(
                self.input_dim, device=self.A.device
            )  # add regularization here so that it's not double-counted under distributed training
        ).contiguous()
        self._coefs = torch.matmul(self.inv_A, self.b)
        self.coefs_valid_for_A = self.gamma * self.A.clone()

        # reset buffers to zero
        self.cur_A.zero_()
        self.cur_b.zero_()
        self.cur_num_obs.zero_()

    def calculate_coefs_if_necessary(self) -> torch.Tensor:
        if not (self.coefs_valid_for_A == self.A).all() or (
            torch.abs(self.cur_A).max().item() > 0
        ):
            # re-calculate the coefs only if the previous value is invalid
            self._calculate_coefs()
        return self._coefs

    @property
    def coefs(self) -> torch.Tensor:
        return self.calculate_coefs_if_necessary()

    def _forward_no_coefs_check(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        # perform forward pass without checking if the current coefficient estimate is still valid
        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha

        mu = torch.matmul(inp, self._coefs)

        if ucb_alpha != 0:
            return mu + ucb_alpha * torch.sqrt(batch_quadratic_form(inp, self.inv_A))
        else:
            return mu

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward can return the mean or a UCB. If returning UCB, the CI width is stddev*ucb_alpha
        If ucb_alpha is not passed in, a fixed alpha from init is used
        """
        self.calculate_coefs_if_necessary()
        return self._forward_no_coefs_check(inp, ucb_alpha)
