#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.training.world_model.compress_model_trainer import CompressModelTrainer
from reagent.training.world_model.seq2reward_trainer import get_Q
from reagent.types import MemoryNetworkInput


logger = logging.getLogger(__name__)


class CompressModelEvaluator:
    def __init__(self, trainer: CompressModelTrainer) -> None:
        self.trainer = trainer
        self.compress_model_network = self.trainer.compress_model_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def evaluate(self, eval_batch: MemoryNetworkInput):
        prev_mode = self.compress_model_network.training
        self.compress_model_network.eval()
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

        self.compress_model_network.train(prev_mode)
        return (detached_loss, q_values, action_distribution)
