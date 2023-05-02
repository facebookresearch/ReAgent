import copy
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from reagent.core.types import CBInput
from reagent.core.utils import get_rank
from reagent.evaluation.cb.utils import zero_out_skipped_obs_weights

logger = logging.getLogger(__name__)


class BaseOfflineEval(torch.nn.Module, ABC):
    """
    Base class for Contextual Bandit Offline Evaluation algorithms. All algorihtms support evaluation of non-stationary
        policies, as required for exploration-exploitation.
    """

    metric_prefix: str = "[model]Offline_Eval_"
    sum_weight_accepted: torch.Tensor
    sum_weight_accepted_local: torch.Tensor
    sum_weight_all_data: torch.Tensor
    sum_weight_all_data_local: torch.Tensor
    sum_weight_since_update_local: torch.Tensor
    num_eval_model_updates: torch.Tensor
    sum_reward_weighted_accepted: torch.Tensor
    sum_reward_weighted_accepted_local: torch.Tensor
    sum_reward_weighted_all_data_local: torch.Tensor
    sum_size_weighted_accepted_local: torch.Tensor
    sum_size_weighted_all_data_local: torch.Tensor
    frac_accepted: torch.Tensor
    avg_reward_accepted: torch.Tensor
    avg_reward_rejected: torch.Tensor
    avg_size_accepted: torch.Tensor
    avg_size_rejected: torch.Tensor
    accepted_rejected_reward_ratio: torch.Tensor
    avg_reward_all_data: torch.Tensor

    def __init__(
        self,
        eval_model: torch.nn.Module,
        logger: Optional[LightningLoggerBase] = None,
    ):
        """
        Initialize the evaluator. The evaluated model is passed in as an input and copied to freeze its state.
        The state of the model remains frozen until method update_eval_model() is called.
        """
        super().__init__()
        self.eval_model = copy.deepcopy(eval_model)
        self.logger = logger

        # below we register the buffers used to maintain the state of Offline Eval
        # Buffers ending in "_local" track per-instance values. Buffers without "_local" track global values aggregated across all instances.
        # At each step we update only the local buffers. The global values are updated periodically and local values are reset to zero at that time.
        # "accepted" means the same as "used"

        # total weight for accepted observations
        self.register_buffer("sum_weight_accepted", torch.zeros(1, dtype=torch.float))
        self.register_buffer(
            "sum_weight_accepted_local", torch.zeros(1, dtype=torch.float)
        )

        # total weight for all observaions (accepted + rejected)
        self.register_buffer("sum_weight_all_data", torch.zeros(1, dtype=torch.float))
        self.register_buffer(
            "sum_weight_all_data_local", torch.zeros(1, dtype=torch.float)
        )

        # total weight processed since last model update and local buffer aggregation
        self.register_buffer(
            "sum_weight_since_update_local", torch.zeros(1, dtype=torch.float)
        )

        # number of times the evaluated model was updated, local metrics aggregated and global metrics logged
        self.register_buffer("num_eval_model_updates", torch.zeros(1, dtype=torch.int))

        # sum of reward*weight for accepted observations
        self.register_buffer(
            "sum_reward_weighted_accepted", torch.zeros(1, dtype=torch.float)
        )
        self.register_buffer(
            "sum_reward_weighted_accepted_local", torch.zeros(1, dtype=torch.float)
        )

        # sum of reward*weight for all observations
        self.register_buffer(
            "sum_reward_weighted_all_data_local", torch.zeros(1, dtype=torch.float)
        )

        # sum of slate_size*weight for accepted observations
        self.register_buffer(
            "sum_size_weighted_accepted_local", torch.zeros(1, dtype=torch.float)
        )

        # sum of slate_size*weight for all observations
        self.register_buffer(
            "sum_size_weighted_all_data_local", torch.zeros(1, dtype=torch.float)
        )

        # fraction of data points which were accepted
        self.register_buffer("frac_accepted", torch.zeros(1, dtype=torch.float))

        # average reward for accepted observations
        self.register_buffer("avg_reward_accepted", torch.zeros(1, dtype=torch.float))

        # average reward for rejected observations
        self.register_buffer("avg_reward_rejected", torch.zeros(1, dtype=torch.float))

        # average slate size for accepted observations
        self.register_buffer("avg_size_accepted", torch.zeros(1, dtype=torch.float))

        # average slate size for rejected observations
        self.register_buffer("avg_size_rejected", torch.zeros(1, dtype=torch.float))

        # ratio of accepted and rejected average rewards
        self.register_buffer(
            "accepted_rejected_reward_ratio", torch.zeros(1, dtype=torch.float)
        )

        # average reward of all data (accepted and rejected)
        self.register_buffer("avg_reward_all_data", torch.zeros(1, dtype=torch.float))

    def ingest_batch(
        self,
        batch: CBInput,
        model_actions: torch.Tensor,
    ) -> CBInput:
        """
        Ingest the batch of data and:
        1. Call self._process_all_data() and self._process_used_data() methods
        2. Modify the batch, zeroing out the weights for observations in which the logged and model actions don't match.

        TODO: support more general logic for zero-ing out the weights (e.g. as required by Doubly Robust - Non-Stationary)
        TODO: remove rows instead of zero-ing out weights (to speed up processing)

        Inputs:
            batch: A batch of training data
            model_actions: A tensor of actions chosen by the evaluated model
        """
        self._process_all_data(batch)
        new_batch = zero_out_skipped_obs_weights(batch, model_actions)
        self._process_used_data(new_batch)
        return new_batch

    @abstractmethod
    def _process_all_data(
        self,
        batch: CBInput,
    ) -> None:
        """
        Process all observations, including the ones where logged action doesn't match the model action. For some algorihtms
            this will be a no-op.
        """
        pass

    @abstractmethod
    def _process_used_data(
        self,
        batch: CBInput,
    ) -> None:
        """
        Process the observations for which the logged action matches the model action. All other observations
            were previously removed (weights wero zero-ed out) by zero_out_skipped_obs_weights()
        """
        pass

    @abstractmethod
    def _aggregate_across_instances(self) -> None:
        """
        Aggregate local data across all instances of the evaluator.
        Used for distributed training.
        """
        pass

    @abstractmethod
    def get_avg_reward(self) -> float:
        """
        Get the current estimate of average reward
        """
        pass

    def update_eval_model(self, eval_model: torch.nn.Module) -> None:
        """
        Update the evaluated model. When exactly to call this is decided by the user and should mimic when
            the model would get updated in a real deployment.
        The model is cloned and set to eval mode.
        """
        self.eval_model = copy.deepcopy(eval_model).eval()

    def attach_logger(self, logger: LightningLoggerBase) -> None:
        """
        Attach a logger to the evaluator. This method is useful in cases where logger is
            not yet available at initialization.
        """
        self.logger = logger

    def log_metrics(self, step: Optional[int] = None) -> None:
        if get_rank() == 0:
            # only log from the main process
            logger.info(self.get_formatted_result_string())
            logger_ = self.logger
            if logger_ is not None:
                metric_dict = {
                    ### cumulative sum and average metrics ###
                    f"{self.metric_prefix}avg_reward": self.get_avg_reward(),
                    f"{self.metric_prefix}sum_weight_accepted": self.sum_weight_accepted.item(),
                    f"{self.metric_prefix}sum_weight_all_data": self.sum_weight_all_data.item(),
                    f"{self.metric_prefix}num_eval_model_updates": self.num_eval_model_updates.item(),
                    ### average values since last logging ###
                    f"{self.metric_prefix}frac_accepted": self.frac_accepted.item(),
                    f"{self.metric_prefix}avg_reward_accepted": self.avg_reward_accepted.item(),
                    f"{self.metric_prefix}avg_reward_rejected": self.avg_reward_rejected.item(),
                    f"{self.metric_prefix}avg_size_accepted": self.avg_size_accepted.item(),
                    f"{self.metric_prefix}avg_size_rejected": self.avg_size_rejected.item(),
                    f"{self.metric_prefix}accepted_rejected_reward_ratio": self.accepted_rejected_reward_ratio.item(),
                    f"{self.metric_prefix}avg_reward_all_data": self.avg_reward_all_data.item(),
                }
                logger_.log_metrics(metric_dict, step=step)

    def get_formatted_result_string(self) -> str:
        return f"Avg reward {self.get_avg_reward():0.3f} based on {int(self.sum_weight_accepted.item())} processed observations (out of {int(self.sum_weight_all_data.item())} observations). The eval model has been updated {self.num_eval_model_updates.item()} times"
