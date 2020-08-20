#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.core.types import MemoryNetworkInput
from reagent.training.world_model.compress_model_trainer import CompressModelTrainer


logger = logging.getLogger(__name__)


class CompressModelEvaluator:
    def __init__(self, trainer: CompressModelTrainer) -> None:
        self.trainer = trainer
        self.compress_model_network = self.trainer.compress_model_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def evaluate(self, eval_tdp: MemoryNetworkInput):
        prev_mode = self.compress_model_network.training
        self.compress_model_network.eval()
        loss = self.trainer.get_loss(eval_tdp)
        detached_loss = loss.cpu().detach().item()
        self.compress_model_network.train(prev_mode)
        return detached_loss
