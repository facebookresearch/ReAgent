#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import reagent.types as rlt
import torch
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer, get_Q


logger = logging.getLogger(__name__)


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

        if self.trainer.view_q_value:
            q_values = (
                get_Q(
                    self.trainer.seq2reward_network, eval_batch, self.trainer.all_permut
                )
                .cpu()
                .mean(0)
                .tolist()
            )
        else:
            q_values = [0] * len(self.trainer.params.action_names)

        self.reward_net.train(reward_net_prev_mode)
        return (detached_loss, q_values)
