#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import torch
from pytorch_lightning.utilities.distributed import ReduceOp, sync_ddp_if_available
from reagent.models.cb_base_model import UCBBaseModel


logger = logging.getLogger(__name__)


def batch_quadratic_form(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute the quadratic form x^T * A * x for a batched input x.
    Inspired by https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
    This is a vectorized implementation of out[i] = x[i].t() @ A @ x[i]
    x shape: (Batch, Feature_dim) or (Batch, Arm, Feature_dim)
    A shape: (Feature_dim, Feature_dim)
    output shape: (Batch) or (Batch, Arm)
    """
    return (torch.matmul(x, A) * x).sum(-1)


def reduce_avg(
    avg_val: torch.Tensor,
    sum_weight: torch.Tensor,
    cur_distributed_avg_val: torch.Tensor,
    cur_distributed_sum_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Get the new weighted average value of a tensor. Steps:
    1. Sum across all trainers the weighted sum of values. This is implemented as a sum of product of
         current-epoch weighted average value and total weight.
    2. Get the new weighted average value by dividing the total weighted sum by the total weight.
        - The total are a sum of the current-epoch values across all trainers and the values from previous
            epochs stored in `avg_val` and `sum_weight`.

    Args:
        avg_val: Current weighted average value (from previous epochs).
        sum_weight: Total weight (from previous epochs).
        cur_distributed_avg_val: Current weighted average value in each trainers in current epoch.
        cur_distributed_sum_weight: Total weight in each trainer in current epoch.
    Returns:
        A new weighted average value.
    """
    total_weight = (
        sync_ddp_if_available(
            cur_distributed_sum_weight.clone(), reduce_op=ReduceOp.SUM
        )
        + sum_weight
    )  # clone the tensor, so that it's not modified in-place
    return (
        avg_val * sum_weight
        + sync_ddp_if_available(
            cur_distributed_avg_val * cur_distributed_sum_weight, reduce_op=ReduceOp.SUM
        )
    ) / total_weight


class LinearRegressionUCB(UCBBaseModel):
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
        super().__init__(input_dim=input_dim)

        self.ucb_alpha = ucb_alpha
        self.l2_reg_lambda = l2_reg_lambda
        self.gamma = gamma
        assert self.gamma <= 1.0 and self.gamma > 0.0

        """
        the buffers below are split between "all data" and "current epoch" values. This is done
          to enable distributed training. "current epoch" values get reduced acorss all trainers at
          the end of an epoch (technically, whenever we estimate the coefficients, which could sometimes
          happen multiple times per epoch) and the sum gets redcuced with "all data" values and the new "all data"
          value is set to the reduction of "all data" and "current epoch" values.
        """
        # A is weighted average of X^T*X across all data
        self.register_buffer("avg_A", torch.zeros(self.input_dim, self.input_dim))
        # b is weighted average of reward*X across all data
        self.register_buffer("avg_b", torch.zeros(self.input_dim))
        # A is weighted average of X^T*X across current epoch
        self.register_buffer("cur_avg_A", torch.zeros(self.input_dim, self.input_dim))
        # b is weighted average of reward*X across current epoch
        self.register_buffer("cur_avg_b", torch.zeros(self.input_dim))

        self.register_buffer("_coefs", torch.zeros(self.input_dim))
        self.register_buffer("inv_avg_A", torch.zeros(self.input_dim, self.input_dim))
        self.register_buffer(
            "coefs_valid_for_avg_A", -torch.ones((self.input_dim, self.input_dim))
        )  # value of avg_A matrix for which self.coefs were estimated
        self.register_buffer("num_obs", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("cur_num_obs", torch.zeros(1, dtype=torch.int64))
        # initialize sum of weights below at small values to avoid dividing by 0
        self.register_buffer("sum_weight", 1e-5 * torch.ones(1, dtype=torch.float))
        self.register_buffer("cur_sum_weight", 1e-5 * torch.ones(1, dtype=torch.float))

        # add a dummy parameter so that DDP doesn't compain about lack of parameters with gradient
        self.dummy_param = torch.nn.parameter.Parameter(torch.zeros(1))

    def _calculate_coefs(self) -> None:
        """
        Compute current estimate of regression coefficients and A_inv=A**-1
        We save both coefficients and A_inv in case they are needed again to avoid recomputing the inverse.
        The coefficients are computed only when needed because their computation can be expensive
            (involves matrix inversion)
        """
        # reduce the values of `avg_A` and `avg_b` across all trainer processes and reduce them with previous-epoch values.
        # The coefficients can't be averaged across trainers because they are a non-linear
        #   function of `A` and `b`
        self.avg_A = reduce_avg(
            self.avg_A, self.sum_weight, self.cur_avg_A, self.cur_sum_weight
        )
        self.avg_b = reduce_avg(
            self.avg_b, self.sum_weight, self.cur_avg_b, self.cur_sum_weight
        )
        self.num_obs += sync_ddp_if_available(self.cur_num_obs, reduce_op=ReduceOp.SUM)
        self.sum_weight += sync_ddp_if_available(
            self.cur_sum_weight, reduce_op=ReduceOp.SUM
        )

        self.inv_avg_A = torch.linalg.pinv(
            self.avg_A
            + self.l2_reg_lambda
            * torch.eye(self.input_dim, device=self.avg_A.device)
            / self.sum_weight  # add regularization here so that it's not double-counted under distributed training
        ).contiguous()
        self._coefs = torch.matmul(self.inv_avg_A, self.avg_b)
        self.coefs_valid_for_avg_A = self.avg_A.clone()

        # reset buffers to zero
        self.cur_avg_A.zero_()
        self.cur_avg_b.zero_()
        self.cur_num_obs.zero_()
        self.cur_sum_weight.zero_()

    def calculate_coefs_if_necessary(self) -> torch.Tensor:
        if not (self.coefs_valid_for_avg_A == self.avg_A).all() or (
            torch.abs(self.cur_avg_A).max().item() > 0
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
            return mu + ucb_alpha * torch.sqrt(
                batch_quadratic_form(inp, self.inv_avg_A) / self.sum_weight
            )
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

    def forward_inference(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        # Don't call the coefficient check if using inference mode
        # because JIT doesn't support the "if" statement
        return self._forward_no_coefs_check(inp, ucb_alpha)
