#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.core.tracker import observable
from reagent.core.types import MemoryNetworkInput
from reagent.training.world_model.compress_model_trainer import CompressModelTrainer
from reagent.training.world_model.seq2reward_trainer import get_Q


logger = logging.getLogger(__name__)


@observable(
    mse_loss=torch.Tensor,
    q_values=torch.Tensor,
    action_distribution=torch.Tensor,
    accuracy=torch.Tensor,
)
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
        mse, acc = self.trainer.get_loss(eval_batch)
        detached_loss = mse.cpu().detach().item()
        acc = acc.item()

        state_first_step = eval_batch.state.float_features[0]
        # shape: batch_size, action_dim
        q_values_all_action_all_data = get_Q(
            self.trainer.seq2reward_network,
            state_first_step,
            self.trainer.all_permut,
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

        # pyre-fixme[16]: `CompressModelEvaluator` has no attribute
        #  `notify_observers`.
        self.notify_observers(
            mse_loss=detached_loss,
            q_values=[q_values],
            action_distribution=[action_distribution],
            accuracy=acc,
        )

        return (detached_loss, q_values, action_distribution, acc)
