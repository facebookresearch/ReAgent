import logging

import torch
from pytorch_lightning.utilities.distributed import ReduceOp, sync_ddp_if_available
from reagent.core.types import CBInput
from reagent.evaluation.cb.base_evaluator import BaseOfflineEval


logger = logging.getLogger(__name__)


EPSILON = 1e-9


class PolicyEvaluator(BaseOfflineEval):
    """
    An offline evaluator for Contextual Bandits, based on the paper https://arxiv.org/pdf/1003.0146.pdf (Algorithm 3)
    """

    sum_reward_weighted: torch.Tensor
    sum_reward_weighted_local: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("sum_reward_weighted", torch.zeros(1, dtype=torch.float))
        self.register_buffer(
            "sum_reward_weighted_local", torch.zeros(1, dtype=torch.float)
        )

    @torch.no_grad()
    def _process_all_data(self, batch: CBInput) -> None:
        if batch.weight is not None:
            self.all_data_sum_weight_local += batch.weight.sum()
        else:
            self.all_data_sum_weight_local += len(batch)

    @torch.no_grad()
    def _process_used_data(self, batch: CBInput) -> None:
        """
        Process the observations for which the logged action matches the model action:
            - Update the average reward
            - Update the total weight counter
        """
        assert batch.reward is not None
        assert batch.weight is not None
        assert batch.weight.shape == batch.reward.shape
        self.sum_reward_weighted_local += (batch.weight * batch.reward).sum()
        self.sum_weight_local += batch.weight.sum()

    def _aggregate_across_instances(self) -> None:
        # sum local values across all trainers, add to the global value
        # clone the tensors to avoid modifying them inplace
        self.sum_reward_weighted += sync_ddp_if_available(
            self.sum_reward_weighted_local.clone(), reduce_op=ReduceOp.SUM
        )
        self.sum_weight += sync_ddp_if_available(
            self.sum_weight_local.clone(), reduce_op=ReduceOp.SUM
        )
        self.all_data_sum_weight += sync_ddp_if_available(
            self.all_data_sum_weight_local.clone(), reduce_op=ReduceOp.SUM
        )
        # reset local values to zero
        self.sum_reward_weighted_local.zero_()
        self.sum_weight_local.zero_()
        self.all_data_sum_weight_local.zero_()

    def get_avg_reward(self) -> float:
        assert (
            self.sum_weight_local.item() == 0.0
        ), f"Non-zero local weight {self.sum_weight_local.item()} in the evaluator. _aggregate_across_instances() Should have beed called to aggregate across all instances and zero-out the local values."
        # return the average reward
        return (self.sum_reward_weighted / (self.sum_weight + EPSILON)).item()
