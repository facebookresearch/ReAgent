#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import final, Optional

import pytorch_lightning as pl

import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.base_evaluator import BaseOfflineEval


logger = logging.getLogger(__name__)


class BaseCBTrainerWithEval(ABC, pl.LightningModule):
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
    # pyre-fixme[14]: `training_step` overrides method defined in `LightningModule`
    #  inconsistently.
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
                # TODO: support distributed training by aggregating sum_weight across trainers.
                if (
                    eval_module.sum_weight_since_update.item()
                    >= eval_model_update_critical_weight
                ):
                    logger.info(
                        f"Updating the evaluated model after {eval_module.sum_weight_since_update.item()} observations"
                    )
                    logger.info(eval_module.get_formatted_result_string())
                    eval_module.update_eval_model(self.scorer)
                    eval_module.sum_weight_since_update.zero_()
                    eval_module.num_eval_model_updates += 1
            with torch.no_grad():
                eval_scores = eval_module.eval_model(batch.context_arm_features)
                model_actions = torch.argmax(eval_scores, dim=1).reshape(-1, 1)
            new_batch = eval_module.ingest_batch(batch, model_actions)
            eval_module.sum_weight_since_update += (
                batch.weight.sum() if batch.weight is not None else len(batch)
            )
        else:
            new_batch = batch
        return self.cb_training_step(new_batch, batch_idx, optimizer_idx)

    def on_train_epoch_end(self):
        eval_module = self.eval_module  # assign to local var to keep pyre happy
        if eval_module is not None:
            logger.info(eval_module.get_formatted_result_string())
