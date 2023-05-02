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

    @torch.no_grad()
    def _process_all_data(self, batch: CBInput) -> None:
        assert batch.reward is not None
        if batch.weight is not None:
            weights = batch.weight
        else:
            weights = torch.ones_like(batch.reward)
        self.sum_weight_all_data_local += weights.sum()
        self.sum_reward_weighted_all_data_local += (weights * batch.reward).sum()
        if batch.arm_presence is not None:
            sizes = batch.arm_presence.sum(1)
        else:
            # assume that all arms are present
            sizes = torch.ones_like(batch.reward) * batch.context_arm_features.shape[1]
        self.sum_size_weighted_all_data_local += (weights.squeeze() * sizes).sum()

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
        # rejected observations have 0 weights, so they get filtered out when we multiply by weight
        self.sum_reward_weighted_accepted_local += (batch.weight * batch.reward).sum()
        self.sum_weight_accepted_local += batch.weight.sum()
        if batch.arm_presence is not None:
            sizes = batch.arm_presence.sum(1)
        else:
            # assume that all arms are present
            sizes = torch.ones_like(batch.reward) * batch.context_arm_features.shape[1]
        self.sum_size_weighted_accepted_local += (batch.weight.squeeze() * sizes).sum()

    def _aggregate_across_instances(self) -> None:
        # sum local values across all trainers, add to the global value
        # clone the tensors to avoid modifying them inplace

        sum_weight_accepted = sync_ddp_if_available(
            self.sum_weight_accepted_local.clone(), reduce_op=ReduceOp.SUM
        )

        sum_weight_all_data = sync_ddp_if_available(
            self.sum_weight_all_data_local.clone(), reduce_op=ReduceOp.SUM
        )

        sum_weight_rejected = sum_weight_all_data - sum_weight_accepted

        sum_reward_weighted_accepted = sync_ddp_if_available(
            self.sum_reward_weighted_accepted_local.clone(), reduce_op=ReduceOp.SUM
        )

        sum_reward_weighted_all_data = sync_ddp_if_available(
            self.sum_reward_weighted_all_data_local.clone(), reduce_op=ReduceOp.SUM
        )
        sum_reward_weighted_rejected = (
            sum_reward_weighted_all_data - sum_reward_weighted_accepted
        )

        sum_size_weighted_accepted = sync_ddp_if_available(
            self.sum_size_weighted_accepted_local.clone(), reduce_op=ReduceOp.SUM
        )
        sum_size_weighted_all_data = sync_ddp_if_available(
            self.sum_size_weighted_all_data_local.clone(), reduce_op=ReduceOp.SUM
        )
        sum_size_weighted_rejected = (
            sum_size_weighted_all_data - sum_size_weighted_accepted
        )

        # udpate the global cumulative sum buffers
        self.sum_reward_weighted_accepted += sum_reward_weighted_accepted
        self.sum_weight_accepted += sum_weight_accepted
        self.sum_weight_all_data += sum_weight_all_data

        # calcualte the metrics for window (since last aggregation across instances)
        self.frac_accepted = sum_weight_accepted / sum_weight_all_data
        self.avg_reward_accepted = sum_reward_weighted_accepted / sum_weight_accepted
        self.avg_reward_rejected = sum_reward_weighted_rejected / sum_weight_rejected

        self.avg_reward_all_data = sum_reward_weighted_all_data / sum_weight_all_data

        self.accepted_rejected_reward_ratio = (
            self.avg_reward_accepted / self.avg_reward_rejected
        )

        self.avg_size_accepted = sum_size_weighted_accepted / sum_weight_accepted

        self.avg_size_rejected = sum_size_weighted_rejected / sum_weight_rejected

        # reset local values to zero
        self.sum_reward_weighted_accepted_local.zero_()
        self.sum_reward_weighted_all_data_local.zero_()
        self.sum_weight_accepted_local.zero_()
        self.sum_weight_all_data_local.zero_()
        self.sum_size_weighted_accepted_local.zero_()
        self.sum_size_weighted_all_data_local.zero_()

    def get_avg_reward(self) -> float:
        assert (
            self.sum_weight_accepted_local.item() == 0.0
        ), f"Non-zero local weight {self.sum_weight_appected_local.item()} in the evaluator. _aggregate_across_instances() Should have beed called to aggregate across all instances and zero-out the local values."
        # return the average reward
        return (
            self.sum_reward_weighted_accepted / (self.sum_weight_accepted + EPSILON)
        ).item()
