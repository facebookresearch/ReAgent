import copy
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.utils import zero_out_skipped_obs_weights
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class BaseOfflineEval(torch.nn.Module, ABC):
    """
    Base class for Contextual Bandit Offline Evaluation algorithms. All algorihtms support evaluation of non-stationary
        policies, as required for exploration-exploitation.

    IMPORTANT: current implementation doesn't support distributed training, only use it with a single training instance.
    """

    sum_weight: torch.Tensor
    all_data_sum_weight: torch.Tensor

    def __init__(
        self,
        eval_model: torch.nn.Module,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Initialize the evaluator. The evaluated model is passed in as an input and copied to freeze its state.
        The state of the model remains frozen until method update_eval_model() is called.
        """
        super().__init__()
        self.eval_model = copy.deepcopy(eval_model)
        self.summary_writer = summary_writer
        self.register_buffer("sum_weight", torch.zeros(1, dtype=torch.float))
        self.register_buffer("all_data_sum_weight", torch.zeros(1, dtype=torch.float))
        self.register_buffer(
            "sum_weight_since_update", torch.zeros(1, dtype=torch.float)
        )
        self.register_buffer("num_eval_model_updates", torch.zeros(1, dtype=torch.int))

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
    def get_avg_reward(self) -> float:
        """
        Get the current estimate of average reward
        """
        pass

    def update_eval_model(self, eval_model: torch.nn.Module) -> None:
        """
        Update the evaluated model. When exactly to call this is decided by the user and should mimic when
            the model would get updated in a real deployment.
        """
        self.eval_model = copy.deepcopy(eval_model)

    def attach_summary_writer(self, summary_writer: SummaryWriter) -> None:
        """
        Attach a SummaryWriter to the evaluator. This method is useful in cases where SummaryWriter is
            not yet available at initialization.
        """
        self.summary_writer = summary_writer

    def log_metrics(self, global_step: Optional[int] = None) -> None:
        logger.info(self.get_formatted_result_string())
        summary_writer = self.summary_writer
        if summary_writer is not None:
            metric_dict = {
                "avg_reward": self.get_avg_reward(),
                "sum_weight": self.sum_weight.item(),
                "all_data_sum_weight": self.all_data_sum_weight.item(),
                "num_eval_model_updates": self.num_eval_model_updates.item(),
            }
            summary_writer.add_scalars(
                "Offline_Eval", metric_dict, global_step=global_step
            )

    def get_formatted_result_string(self) -> str:
        return f"Avg reward {self.get_avg_reward():0.3f} based on {int(self.sum_weight.item())} processed observations (out of {int(self.all_data_sum_weight.item())} observations). The eval model has been updated {self.num_eval_model_updates.item()} times"
