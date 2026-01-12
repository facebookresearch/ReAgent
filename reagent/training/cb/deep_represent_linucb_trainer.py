#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging

import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB
from reagent.training.cb.linucb_trainer import LinUCBTrainer
from reagent.training.cb.supervised_trainer import LOSS_TYPES

logger = logging.getLogger(__name__)


class DeepRepresentLinUCBTrainer(LinUCBTrainer):
    """
    The trainer for a Contextual Bandit model, where deep represent layer serves as feature processor,
    and then processed features are fed to LinUCB layer to produce UCB score.
    This is extension of LinUCBTrainer. More details refer to docstring of LinUCBTrainer.

    Reference:
    - LinUCB : https://arxiv.org/pdf/1003.0146.pdf
    - DeepRepresentLinUCB : https://arxiv.org/pdf/2012.01780.pdf
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_type: str = "mse",  # one of the LOSS_TYPES names
        **kwargs,
    ):
        super().__init__(
            automatic_optimization=True,
            policy=policy,
            **kwargs,
        )
        assert isinstance(policy.scorer, DeepRepresentLinearRegressionUCB), (
            "Trainer requires the policy scorer to be DeepRepresentLinearRegressionUCB"
        )
        self.scorer = policy.scorer
        self.loss_fn = LOSS_TYPES[loss_type]
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.scorer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        """
        Here scorer (DeepRepresentLinearRegressionUCB) outputs Dict
            {
                "pred_label": pred_label,
                "pred_sigma": pred_sigma,
                "ucb": ucb,
                "mlp_out_with_ones": mlp_out_with_ones,
            }
        The pred_label is useful for calculating MSE loss.
        The mlp_out_with_ones is useful for updating LinUCB parameters (A, b, etc)
        """
        model_output = self.scorer(inp=batch.features_of_chosen_arm)  # noqa
        # this calls scorer.forward() so as to update pred_u, and to grad descent on deep_represent module

        pred_label, mlp_out_with_ones = (
            model_output["pred_label"],
            model_output["mlp_out_with_ones"],
        )

        assert batch.label is not None  # to satisfy Pyre
        label = batch.label.squeeze(-1)
        assert pred_label.shape == label.shape, (
            f"Shapes of model prediction {pred_label.shape} and label {label.shape} have to match"
        )
        # compute the NN loss
        # weighted average loss
        losses = self.loss_fn(pred_label, label, reduction="none")
        weight = batch.effective_weight
        loss = (losses * weight.squeeze(-1)).sum() / losses.shape[0]

        # update LinUCB parameters
        self.update_params(mlp_out_with_ones.detach(), batch.label, weight)

        return loss
