#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import enum
import logging
from typing import Optional

import reagent.core.parameters as rlp
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.dataclasses import field
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.rl_trainer_pytorch import RLTrainerMixin

logger = logging.getLogger(__name__)


class NextSlateValueNormMethod(enum.Enum):
    """
    The Q value of the current slate item is the sum of the item's short-term reward and
    the normalized sum of all item Q-values on the next slate.
    We can normalize the sum by either the current slate size (NORM_BY_CURRENT_SLATE_SIZE)
    or the next slate size (NORM_BY_NEXT_SLATE_SIZE).
    This enum distinguishes between these two different ways of normalizing the next slate value.
    """

    NORM_BY_CURRENT_SLATE_SIZE = "norm_by_current_slate_size"
    NORM_BY_NEXT_SLATE_SIZE = "norm_by_next_slate_size"


class SlateQTrainer(RLTrainerMixin, ReAgentLightningModule):
    def __init__(
        self,
        q_network,
        q_network_target,
        slate_size,
        # Start SlateQTrainerParameters
        rl: rlp.RLParameters = field(  # noqa: B008
            default_factory=lambda: rlp.RLParameters(maxq_learning=False)
        ),
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        slate_opt_parameters: Optional[rlp.SlateOptParameters] = None,
        discount_time_scale: Optional[float] = None,
        single_selection: bool = True,
        next_slate_value_norm_method: NextSlateValueNormMethod = NextSlateValueNormMethod.NORM_BY_CURRENT_SLATE_SIZE,
        minibatch_size: int = 1024,
        evaluation: rlp.EvaluationParameters = field(  # noqa: B008
            default_factory=lambda: rlp.EvaluationParameters(calc_cpe_in_training=False)
        ),
    ) -> None:
        """
        Args:
            q_network: states, action -> q-value
            slate_size(int): a fixed slate size
            rl (optional): an instance of the RLParameter class, which
                defines relevant hyperparameters
            optimizer (optional): the optimizer class and
                optimizer hyperparameters for the q network(s) optimizer
            discount_time_scale (optional): use to control the discount factor (gamma)
                relative to the time difference (t2-t1), i.e., gamma^((t2-t1)/time_scale).
                If it is absent, we won't adjust the discount factor by the time difference.
            single_selection (optional): TBD
            next_slate_value_norm_method (optional): how to calculate the next slate value
                when single_selection is False. By default we use NORM_BY_CURRENT_SLATE_SIZE.
            minibatch_size (optional): the size of the minibatch
            evaluation (optional): TBD
        """
        super().__init__()
        self.rl_parameters = rl

        self.discount_time_scale = discount_time_scale
        self.single_selection = single_selection
        self.next_slate_value_norm_method = next_slate_value_norm_method

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = optimizer

        self.slate_size = slate_size
        self.slate_opt_parameters = slate_opt_parameters

    def configure_optimizers(self):
        optimizers = []

        optimizers.append(
            self.q_network_optimizer.make_optimizer_scheduler(
                self.q_network.parameters()
            )
        )

        target_params = list(self.q_network_target.parameters())
        source_params = list(self.q_network.parameters())
        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def _action_docs(
        self,
        state: rlt.FeatureData,
        action: torch.Tensor,
        terminal_mask: Optional[torch.Tensor] = None,
    ) -> rlt.DocList:
        # for invalid indices, simply set action to 0 so we can batch index still
        if terminal_mask is not None:
            assert terminal_mask.shape == (action.shape[0],), (
                f"{terminal_mask.shape} != 0th dim of {action.shape}"
            )
            action[terminal_mask] = torch.zeros_like(action[terminal_mask])
        docs = state.candidate_docs
        assert docs is not None
        return docs.select_slate(action)

    def _get_unmasked_q_values(
        self, q_network, state: rlt.FeatureData, slate: rlt.DocList
    ) -> torch.Tensor:
        """Gets the q values from the model and target networks"""
        batch_size, slate_size, _ = slate.float_features.shape
        # TODO: Probably should create a new model type
        return q_network(
            state.repeat_interleave(slate_size, dim=0), slate.as_feature_data()
        ).view(batch_size, slate_size)

    @torch.no_grad()
    def _get_maxq_next_action(self, next_state: rlt.FeatureData) -> torch.Tensor:
        """Get the next action list based on the slate optimization strategy."""
        slate_opt_parameters = self.slate_opt_parameters
        assert slate_opt_parameters is not None

        if slate_opt_parameters.method == rlp.SlateOptMethod.TOP_K:
            return self._get_maxq_topk(next_state)
        else:
            raise NotImplementedError(
                "SlateQ with optimization method other than TOP_K is not implemented."
            )

    def _get_maxq_topk(self, next_state: rlt.FeatureData) -> torch.Tensor:
        candidate_docs = next_state.candidate_docs
        assert candidate_docs is not None

        batch_size, num_candidates, _ = candidate_docs.float_features.shape
        assert 0 < self.slate_size <= num_candidates

        docs = candidate_docs.select_slate(
            torch.arange(num_candidates).repeat(batch_size, 1)
        )
        next_q_values = self._get_unmasked_q_values(
            self.q_network_target, next_state, docs
        ) * self._get_docs_value(docs)
        _, next_actions = torch.topk(next_q_values, self.slate_size, dim=1)

        return next_actions

    def _get_docs_value(self, docs: rlt.DocList) -> torch.Tensor:
        # Multiplying by the mask to filter out selected padding items.
        value = docs.value * docs.mask
        if self.single_selection:
            value = F.softmax(value, dim=1)
        return value

    def _get_slate_size(self, state: rlt.FeatureData) -> torch.Tensor:
        """Get the actual size (ignore all padded items) of each slate by summing item masks."""
        mask = self._get_item_mask(state)
        return torch.minimum(
            mask.sum(1, keepdim=True),
            torch.tensor([self.slate_size], device=mask.device),
        )

    def _get_item_mask(self, state: rlt.FeatureData) -> torch.Tensor:
        """Get the mask from the given state."""
        candidate_docs = state.candidate_docs
        assert candidate_docs is not None
        return candidate_docs.mask

    def _get_avg_by_slate_size(self, batch: rlt.SlateQInput):
        """Get the slate_size for averaging the sum of slate value."""
        if (
            self.next_slate_value_norm_method
            == NextSlateValueNormMethod.NORM_BY_NEXT_SLATE_SIZE
        ):
            return self._get_slate_size(batch.next_state)
        if (
            self.next_slate_value_norm_method
            == NextSlateValueNormMethod.NORM_BY_CURRENT_SLATE_SIZE
        ):
            return self._get_slate_size(batch.state)
        raise NotImplementedError(
            f"The next_slate_value_norm_method {self.next_slate_value_norm_method} has not been implemented"
        )

    def train_step_gen(self, training_batch: rlt.SlateQInput, batch_idx: int):
        assert isinstance(training_batch, rlt.SlateQInput), (
            f"learning input is a {type(training_batch)}"
        )

        reward = training_batch.reward
        reward_mask = training_batch.reward_mask

        discount_tensor = torch.full_like(reward, self.gamma)

        # Adjust the discount factor by the time_diff if the discount_time_scale is provided,
        # and the time_diff exists in the training_batch.
        if self.discount_time_scale and training_batch.time_diff is not None:
            discount_tensor = discount_tensor ** (
                training_batch.time_diff / self.discount_time_scale
            )

        next_action = (
            self._get_maxq_next_action(training_batch.next_state)
            if self.rl_parameters.maxq_learning
            else training_batch.next_action
        )

        terminal_mask = (training_batch.not_terminal.to(torch.bool) == False).squeeze(1)
        next_action_docs = self._action_docs(
            training_batch.next_state,
            next_action,
            terminal_mask=terminal_mask,
        )
        next_q_values = torch.sum(
            self._get_unmasked_q_values(
                self.q_network_target,
                training_batch.next_state,
                next_action_docs,
            )
            * self._get_docs_value(next_action_docs),
            dim=1,
            keepdim=True,
        )

        # If not single selection, divide max-Q by the actual slate size.
        if not self.single_selection:
            next_q_values = next_q_values / self._get_avg_by_slate_size(training_batch)

        filtered_max_q_vals = next_q_values * training_batch.not_terminal.float()
        target_q_values = reward + (discount_tensor * filtered_max_q_vals)
        # Don't mask if not single selection
        if self.single_selection:
            target_q_values = target_q_values[reward_mask]

        # Get Q-value of action taken
        action_docs = self._action_docs(training_batch.state, training_batch.action)
        q_values = self._get_unmasked_q_values(
            self.q_network, training_batch.state, action_docs
        )
        if self.single_selection:
            q_values = q_values[reward_mask]

        all_action_scores = q_values.detach()

        value_loss = F.mse_loss(q_values, target_q_values)
        yield value_loss

        if not self.single_selection:
            all_action_scores = all_action_scores.sum(dim=1, keepdim=True)

        # Logging at the end to schedule all the cuda operations first
        self.reporter.log(
            td_loss=value_loss,
            model_values_on_logged_actions=all_action_scores,
        )

        # Use the soft update rule to update the target networks
        result = self.soft_update_result()
        self.log(
            "td_loss", value_loss, prog_bar=True, batch_size=training_batch.batch_size()
        )
        yield result
