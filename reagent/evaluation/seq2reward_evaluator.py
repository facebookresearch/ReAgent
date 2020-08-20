#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.core.types import PreprocessedTrainingBatch
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer


logger = logging.getLogger(__name__)


class Seq2RewardEvaluator:
    def __init__(self, trainer: Seq2RewardTrainer) -> None:
        self.trainer = trainer
        self.reward_net = self.trainer.seq2reward_network

    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch):
        reward_net_prev_mode = self.reward_net.training
        self.reward_net.eval()
        # pyre-fixme[6]: Expected `MemoryNetworkInput` for 1st param but got
        #  `PreprocessedTrainingBatch`.
        loss = self.trainer.compute_loss(eval_tdp)
        detached_loss = loss.cpu().detach().item()
        q_values = (
            self.trainer.get_Q(
                # pyre-fixme[6]: Expected `MemoryNetworkInput` for 1st param but got
                #  `PreprocessedTrainingBatch`.
                eval_tdp,
                eval_tdp.batch_size(),
                self.trainer.params.multi_steps,
                len(self.trainer.params.action_names),
            )
            .mean(0)
            .tolist()
        )
        self.reward_net.train(reward_net_prev_mode)
        return (detached_loss, q_values)

    def finish(self):
        pass
