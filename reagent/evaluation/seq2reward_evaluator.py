#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer
from reagent.types import PreprocessedTrainingBatch


logger = logging.getLogger(__name__)


class Seq2RewardEvaluator:
    def __init__(self, trainer: Seq2RewardTrainer) -> None:
        self.trainer = trainer

    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch):
        reward_net = self.trainer.seq2reward_network
        reward_net_prev_mode = reward_net.training
        reward_net.eval()
        # pyre-fixme[6]: Expected `MemoryNetworkInput` for 1st param but got
        #  `PreprocessedTrainingBatch`.
        loss = self.trainer.get_loss(eval_tdp)
        detached_loss = loss.cpu().detach().item()
        reward_net.train(reward_net_prev_mode)
        return detached_loss
