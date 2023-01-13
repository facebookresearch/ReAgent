#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import final, Optional

import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.base_evaluator import BaseOfflineEval
from reagent.training.reagent_lightning_module import ReAgentLightningModule


logger = logging.getLogger(__name__)


class BaseCBTrainerWithEval(ABC, ReAgentLightningModule):
    """
    The base class for Contextual Bandit models. A minimal implementation of a specific model requires providing a specific
        implementation for only the cb_training_step() method.
    The main functionality implemented in this base class is the integration of Offline Evaluation into the training loop.
    """

    scorer: torch.nn.Module

    def __init__(
        self, eval_model_update_critical_weight: Optional[float] = None, *args, **kwargs
    ):
        """
        Agruments:
            eval_model_update_critical_weight: Maximum total weight of training data (all data, not just that where
                logged and model actions match) after which we update the state of the evaluated model.
        """
        super().__init__(*args, **kwargs)
        self.eval_module: Optional[BaseOfflineEval] = None
        self.eval_model_update_critical_weight = eval_model_update_critical_weight

    def attach_eval_module(self, eval_module: BaseOfflineEval):
        """
        Attach an Offline Evaluation module. It will kleep track of reward during training and filter training batches.
        """
        self.eval_module = eval_module

    @abstractmethod
    def cb_training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        """
        This method impements the actual training step. See training_step() for more details
        """
        pass

    @final
    def training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        """
        This method combines 2 things in order to enable Offline Evaluation of non-stationary CB algorithms:
        1. If offline evaluator is defined, it will pre-process the batch - keep track of the reward and filter out some observations.
        2. The filtered batch will be fed to the cb_training_step() method, which implements the actual training logic.

        DO NOT OVERRIDE THIS METHOD IN SUBCLASSES, IT'S @final. Instead, override cb_training_step().
        """
        eval_module = self.eval_module  # assign to local var to keep pyre happy
        if eval_module is not None:
            # update the model if we've processed enough samples
            eval_model_update_critical_weight = self.eval_model_update_critical_weight
            if eval_model_update_critical_weight is not None:
                if (
                    eval_module.sum_weight_since_update_local.item()
                    >= eval_model_update_critical_weight
                ):
                    logger.info(
                        f"Updating the evaluated model after {eval_module.sum_weight_since_update_local.item()} observations"
                    )
                    eval_module.update_eval_model(self.scorer)
                    eval_module.sum_weight_since_update_local.zero_()
                    eval_module.num_eval_model_updates += 1
                    eval_module._aggregate_across_instances()
                    eval_module.log_metrics(global_step=self.global_step)
            with torch.no_grad():
                eval_scores = eval_module.eval_model(batch.context_arm_features)
                if batch.arm_presence is not None:
                    # mask out non-present arms
                    eval_scores = torch.masked.as_masked_tensor(
                        eval_scores, batch.arm_presence.bool()
                    )
                    model_actions = (
                        # pyre-fixme[16]: `Tensor` has no attribute `get_data`.
                        torch.argmax(eval_scores, dim=1)
                        .get_data()
                        .reshape(-1, 1)
                    )
                else:
                    model_actions = torch.argmax(eval_scores, dim=1).reshape(-1, 1)
            new_batch = eval_module.ingest_batch(batch, model_actions)
            eval_module.sum_weight_since_update_local += (
                batch.weight.sum() if batch.weight is not None else len(batch)
            )
        else:
            new_batch = batch
        return self.cb_training_step(new_batch, batch_idx, optimizer_idx)

    def on_train_epoch_end(self):
        eval_module = self.eval_module  # assign to local var to keep pyre happy
        if eval_module is not None:
            if eval_module.sum_weight_since_update_local.item() > 0:
                # only aggregate if we've processed new data since last aggregation.
                eval_module._aggregate_across_instances()
            eval_module.log_metrics(global_step=self.global_step)
