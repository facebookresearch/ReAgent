#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import final, List, Optional, Union

import torch
from reagent.core.types import CBInput
from reagent.core.utils import get_rank
from reagent.evaluation.cb.base_evaluator import BaseOfflineEval
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from torchrec.metrics.metric_module import RecMetricModule


logger = logging.getLogger(__name__)


def _add_chosen_arm_features(
    batch: Union[CBInput, List[CBInput]]
) -> Union[CBInput, List[CBInput]]:
    """
    Add the features for chosen arms to the batch.
    For joint models:
        - Both input and output are`CBInput` objects
        - batch.all_arm_features is a 3D tensor of shape (batch_size, num_arms, arm_dim)
        - `batch.action` is used to choose which features to extract.
    For joint models:
         - Both input and output are `List[CBInput]` objects of len `num_arms`
         - batch.all_arm_features is a 2D tensor of shape (batch_size, arm_dim)
         - This function just extracts the `all_arm_features` attribute from batch and packages them back into a list

    Args:
        batch: A batch of input data.
            Attributes of the batch:
                all_arm_features: Tensor with features of all available arms.
                action: 2D Tensor of shape (batch_size, 1) with dtype long. For each observation
                    it holds the index of the chosen arm.
    Returns:
        For joint models:
            A 2D Tensor of shape (batch_size, arm_dim) with features of chosen arms.
        For disjoint models:
            A list of 2D Tensors of shape (batch_size, arm_dim)
    """
    if isinstance(batch, CBInput):
        assert batch.context_arm_features.ndim == 3
        assert batch.action is not None
        return replace(
            batch,
            features_of_chosen_arm=torch.gather(
                batch.context_arm_features,
                1,
                batch.action.unsqueeze(-1).expand(
                    -1, 1, batch.context_arm_features.shape[2]
                ),
            ).squeeze(1),
        )
    elif isinstance(batch, list):
        assert isinstance(batch[0], CBInput)
        assert batch[0].context_arm_features.ndim == 2
        return [
            replace(b, features_of_chosen_arm=b.context_arm_features) for b in batch
        ]
    else:
        raise ValueError(
            f"Unexpected input type {type(batch)} for _add_chosen_arm_features"
        )


class BaseCBTrainerWithEval(ABC, ReAgentLightningModule):
    """
    The base class for Contextual Bandit models. A minimal implementation of a specific model requires providing a specific
        implementation for only the cb_training_step() method.
    The main functionality implemented in this base class is the integration of Offline Evaluation into the training loop.
    """

    scorer: torch.nn.Module

    def __init__(
        self,
        eval_model_update_critical_weight: Optional[float] = None,
        recmetric_module: Optional[RecMetricModule] = None,
        log_every_n_steps: int = 0,
        *args,
        **kwargs,
    ):
        """
        Agruments:
            eval_model_update_critical_weight: Maximum total weight of training data (all data, not just that where
                logged and model actions match) after which we update the state of the evaluated model.
        """
        super().__init__(*args, **kwargs)
        self.eval_module: Optional[BaseOfflineEval] = None
        self.eval_model_update_critical_weight = eval_model_update_critical_weight
        self.recmetric_module = recmetric_module
        self.log_every_n_steps = log_every_n_steps
        assert (log_every_n_steps > 0) == (
            recmetric_module is not None
        ), "recmetric_module should be provided if and only if log_every_n_steps > 0"

    def _check_input(self, batch: CBInput, offline_eval: bool = False) -> None:
        """
        Check that the input batch satisfies the following assumptions:
            1. context_arm_features is 3D with dimensions: batch_size, arm_count, feature_dim
            2. Label and action are not none
            3. Batch size is the same for action, reward and context_arm_features
        """
        assert batch.context_arm_features.ndim == 3
        assert batch.label is not None
        assert batch.action is not None
        assert len(batch.action) == len(batch.label)
        assert len(batch.action) == batch.context_arm_features.shape[0]
        if offline_eval:
            assert batch.reward is not None
            assert len(batch.action) == len(batch.reward)

    def attach_eval_module(self, eval_module: BaseOfflineEval) -> None:
        """
        Attach an Offline Evaluation module. It will kleep track of reward during training and filter training batches.
        """
        self.eval_module = eval_module

    @abstractmethod
    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """
        This method impements the actual training step. See training_step() for more details
        """
        pass

    @final
    def training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """
        This method combines 2 things in order to enable Offline Evaluation of non-stationary CB algorithms:
        1. If offline evaluator is defined, it will pre-process the batch - keep track of the reward and filter out some observations.
        2. The filtered batch will be fed to the cb_training_step() method, which implements the actual training logic.

        DO NOT OVERRIDE THIS METHOD IN SUBCLASSES, IT'S @final. Instead, override cb_training_step().
        """
        eval_module = self.eval_module  # assign to local var to keep pyre happy
        offline_eval_enabled = eval_module is not None
        self._check_input(batch, offline_eval=offline_eval_enabled)
        if offline_eval_enabled:
            assert eval_module is not None
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
                    eval_module.log_metrics(step=self.global_step)
            with torch.no_grad():
                model_output = eval_module.eval_model(batch.context_arm_features)
                ucb = model_output["ucb"]
                eval_scores = ucb
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
        new_batch = _add_chosen_arm_features(new_batch)
        ret = self.cb_training_step(new_batch, batch_idx, optimizer_idx)  # pyre-ignore
        if isinstance(new_batch, CBInput):
            # recmetrics currently supported only for joint model
            assert new_batch.features_of_chosen_arm is not None
            self._update_recmetrics(
                new_batch, batch_idx, new_batch.features_of_chosen_arm
            )
        return ret

    def on_train_start(self) -> None:
        # attach logger to the eval module
        eval_module = self.eval_module
        logger = self.logger
        if (eval_module is not None) and (logger is not None) and (get_rank() == 0):
            eval_module.attach_logger(logger)

    def on_train_epoch_end(self) -> None:
        eval_module = self.eval_module  # assign to local var to keep pyre happy
        if eval_module is not None:
            if eval_module.sum_weight_since_update_local.item() > 0:
                # only aggregate if we've processed new data since last aggregation.
                eval_module._aggregate_across_instances()
            eval_module.log_metrics(step=self.global_step)

    def _update_recmetrics(
        self, batch: CBInput, batch_idx: int, x: torch.Tensor
    ) -> None:
        recmetric_module = self.recmetric_module
        if (recmetric_module is not None) and (batch_idx % self.log_every_n_steps == 0):
            # get point predictions (expected value, uncertainty is ignored)
            # this could be expensive because the coefficients have to be computed via matrix inversion
            model_output = self.scorer(x)
            preds = model_output["pred_label"]

            weight = batch.weight
            if weight is None:
                assert batch.label is not None
                weight = torch.ones_like(batch.label)
            recmetric_module.update(
                {
                    "prediction": preds,
                    "label": batch.label,
                    "weight": weight,
                }
            )
            self._log_recmetrics(step=self.global_step)

    def _log_recmetrics(self, step: Optional[int] = None) -> None:
        recmetric_module = self.recmetric_module
        assert recmetric_module is not None
        computation_results = recmetric_module.compute()
        if get_rank() == 0:
            logger_ = self.logger
            assert logger_ is not None
            computation_results = {
                "[model]" + k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in computation_results.items()
            }
            logger_.log_metrics(computation_results, step=step)
            logger.info(f"Logging torchrec metrics {computation_results}")
