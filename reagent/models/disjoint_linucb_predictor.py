#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import torch

from pytorch_lightning.utilities.distributed import ReduceOp, sync_ddp_if_available
from reagent.models.base import ModelBase


logger = logging.getLogger(__name__)


def batch_quadratic_form_multi_arms(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute the quadratic form x^T * A * x for a batched input x. (exploration term in UCB)
    x shape: (B, N)
    A shape: (num_arms, N, N)
    output shape: (B, num_arms)
    B is the batch size above
    N is the featur dimension here
    """
    # xta = x^T A
    # the input x is already transposed, so no transpose is applied below.
    # xta is dimension (num_arms, B, N)
    xta = torch.matmul(x, A)
    # einsum i: arm_idx; j: data index in batch; k: feature index
    return torch.einsum("ijk, jk -> ji", xta, x)


class DisjointLinearRegressionUCB(ModelBase):
    """
    A linear regression model for Disjoint LinUCB.
    Note that instead of being trained by a PyTorch optimizer, we explicitly
        update attributes A and b.

    Args:
        input_dim: Dimension of input data
        l2_reg_lambda: The weight on L2 regularization
        ucb_alpha: The coefficient on the standard deviation in UCB formula.
        gamma: The discount factor to avoid exploding numbers when doing incremental training
        using gamma as part of simplified D-LinUCB: https://arxiv.org/pdf/1909.09146.pdf
        In this simplified version of D-LinUCB, we calculate A = \sum \gamma^t xx^T
        to discount the old data. See N2441818 for why we do this.
        set gamma=1.0 to use traditional model
    """

    def __init__(
        self,
        num_arms: int,
        input_dim: int,
        l2_reg_lambda: float = 1.0,
        ucb_alpha: float = 1.0,
        gamma: float = 1.0,
    ):

        """
        self.A: num_arms * dim * dim
        self.A_inv: num_arms * dim * dim
        self.b: num_arms * dim
        self.coefs: num_arms * dim
        dim is feature dimension
        """
        super().__init__()

        self.num_arms = num_arms
        self.input_dim = input_dim
        self.ucb_alpha = ucb_alpha
        self.gamma = gamma
        assert self.gamma <= 1.0 and self.gamma > 0.0
        self.l2_reg_lambda = l2_reg_lambda
        self.register_buffer(
            "A",
            torch.zeros((self.input_dim, self.input_dim)).repeat(self.num_arms, 1, 1),
        )
        self.register_buffer("b", torch.zeros(self.num_arms, self.input_dim))
        self.register_buffer(
            "coefs", torch.zeros(self.input_dim).repeat(self.num_arms, 1)
        )
        self.register_buffer(
            "inv_A",
            torch.eye(self.input_dim).repeat(self.num_arms, 1, 1),
        )
        self.register_buffer(
            "coefs_valid_for_A",
            -torch.ones((self.input_dim, self.input_dim)).repeat(self.num_arms, 1, 1),
        )  # value of A matrix for which self.coefs were estimated
        # add a dummy parameter so that DDP doesn't compain about lack of parameters with gradient
        self.dummy_param = torch.nn.parameter.Parameter(torch.zeros(1))

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim)

    def _estimate_coefs(self):
        """
        Compute current estimate of regression coefficients and A_inv=A**-1
        We save both coefficients and A_inv in case they are needed again before we add observations

        self.coefs: num_arms * dim
        """
        self.A = sync_ddp_if_available(self.A, reduce_op=ReduceOp.SUM)
        self.b = sync_ddp_if_available(self.b, reduce_op=ReduceOp.SUM)
        device = self.b.device
        # add regularization here so that it's not double-counted under distributed training
        # send them to the same device to avoid errors when doing dpp, example failed job without this: aienv-0d5c64a3b3
        m = self.A.to(device) + self.l2_reg_lambda * torch.eye(self.input_dim).to(
            device
        )
        self.inv_A = torch.inverse(m).contiguous()
        assert self.inv_A.size()[0] == self.b.size()[0]

        # inv_A: (num_arms, d, d)
        # self.b: (num_arms, d)
        # einsum j: arm_idx, d: feature dimension index
        # output self.coefs: (num_arms, d)
        self.coefs = torch.einsum("jkl,jl->jk", self.inv_A, self.b)
        # copy A to make coefs_valid_for_A the same as A
        # need coefs_valid_for_A to check if we have done _estimate_coefs
        # this is needed to avoid redundant re-compute of coefs in forward function.
        self.coefs_valid_for_A = self.gamma * self.A.clone()

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward can return the mean or a UCB. If returning UCB, the CI width is stddev*ucb_alpha
        If ucb_alpha is not passed in, a fixed alpha from init is used

        inp: num_batch * dim
        self.coefs: num_arms * dim
        output: num_batch * num_arms
        output is in the format:
        [
            [score_1, score_2, ..., score_{num_arms}],
            ....,
        ]
        """
        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha
        if not (self.coefs_valid_for_A == self.A).all():
            self._estimate_coefs()

        results = torch.matmul(inp, self.coefs.t())
        if ucb_alpha == 0:
            return results
        results += ucb_alpha * torch.sqrt(
            batch_quadratic_form_multi_arms(inp, self.inv_A)
        )
        return results
