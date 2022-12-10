#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB
from reagent.optimizer.union import Optimizer__Union
from reagent.training.cb.linucb_trainer import _get_chosen_arm_features, LinUCBTrainer

logger = logging.getLogger(__name__)


class DeepRepresentLinUCBTrainer(LinUCBTrainer):
    """
    The trainer for a Contextual Bandit model, where deep represent layer serves as feature processor,
    and then processed features are fed to LinUCB layer to produce UCB score.
    This is extension of LinUCBTrainer. More details refer to docstring of LinUCBTrainer.

    Reference:
    - LinUCB : https://arxiv.org/pdf/2012.01780.pdf
    - DeepRepresentLinUCB : https://arxiv.org/pdf/1003.0146.pdf
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            **kwargs,
        )
        assert isinstance(
            policy.scorer, DeepRepresentLinearRegressionUCB
        ), "Trainer requires the policy scorer to be DeepRepresentLinearRegressionUCB"
        self.scorer = policy.scorer
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = optimizer

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.optimizer.make_optimizer_scheduler(self.scorer.parameters())
        )
        return optimizers

    def cb_training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        self._check_input(batch)
        assert batch.action is not None  # to satisfy Pyre
        assert batch.reward is not None  # to satisfy Pyre
        x = _get_chosen_arm_features(batch.context_arm_features, batch.action)

        pred_ucb = self.scorer(  # noqa
            inp=x
        )  # this calls scorer.forward() so as to update pred_u, and to grad descent on deep_represent module
        loss = self.loss_fn(self.scorer.pred_u, batch.reward.t())

        # update parameters
        self.update_params(self.scorer.mlp_out.detach(), batch.reward, batch.weight)

        return loss
