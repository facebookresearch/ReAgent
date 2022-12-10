import torch
from reagent.core.types import CBInput
from reagent.evaluation.cb.base_evaluator import BaseOfflineEval


EPSILON = 1e-9


class PolicyEvaluator(BaseOfflineEval):
    """
    An offline evaluator for Contextual Bandits, based on the paper https://arxiv.org/pdf/1003.0146.pdf (Algorithm 3)
    """

    avg_reward_weighted: torch.Tensor

    def __init__(self, eval_model: torch.nn.Module):
        super().__init__(eval_model=eval_model)
        self.register_buffer("avg_reward_weighted", torch.zeros(1, dtype=torch.float))

    @torch.no_grad()
    def _process_all_data(self, batch: CBInput) -> None:
        if batch.weight is not None:
            self.all_data_sum_weight += batch.weight.sum()
        else:
            self.all_data_sum_weight += len(batch)

    @torch.no_grad()
    def _process_used_data(self, batch: CBInput) -> None:
        """
        Process the observations for which the logged action matches the model action:
            - Update the average reward
            - Update the total weight counter
        """
        assert batch.reward is not None
        assert batch.weight is not None
        batch_sum_weight = batch.weight.sum()
        assert batch.weight.shape == batch.reward.shape
        self.avg_reward_weighted = (
            self.avg_reward_weighted * self.sum_weight
            + (batch.weight * batch.reward).sum()
        ) / (self.sum_weight + batch_sum_weight + EPSILON)
        self.sum_weight += batch_sum_weight

    def get_avg_reward(self) -> float:
        return self.avg_reward_weighted.item()
