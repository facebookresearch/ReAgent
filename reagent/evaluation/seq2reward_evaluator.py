#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import reagent.types as rlt
import torch
from reagent.core.tracker import observable
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer, get_Q

logger = logging.getLogger(__name__)


@observable(
    mse_loss=torch.Tensor, q_values=torch.Tensor, action_distribution=torch.Tensor
)
class Seq2RewardEvaluator:
    def __init__(self, trainer: Seq2RewardTrainer) -> None:
        self.trainer = trainer
        self.reward_net = self.trainer.seq2reward_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def evaluate(self, eval_batch: rlt.MemoryNetworkInput):
        reward_net_prev_mode = self.reward_net.training
        self.reward_net.eval()
        loss = self.trainer.get_loss(eval_batch)
        detached_loss = loss.cpu().detach().item()

        # shape: batch_size, action_dim
        q_values_all_action_all_data = get_Q(
            self.trainer.seq2reward_network, eval_batch, self.trainer.all_permut
        ).cpu()
        q_values = q_values_all_action_all_data.mean(0).tolist()

        action_distribution = torch.bincount(
            torch.argmax(q_values_all_action_all_data, dim=1),
            minlength=len(self.trainer.params.action_names),
        )
        # normalize
        action_distribution = (
            action_distribution.float() / torch.sum(action_distribution)
        ).tolist()
        # pyre-fixme[16]: `Seq2RewardEvaluator` has no attribute
        #  `notify_observers`.
        self.notify_observers(
            mse_loss=loss,
            q_values=[q_values],
            action_distribution=[action_distribution],
        )

        self.reward_net.train(reward_net_prev_mode)
        return (detached_loss, q_values, action_distribution)
