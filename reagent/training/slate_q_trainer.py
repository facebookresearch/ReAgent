#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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


class SlateQTrainer(RLTrainerMixin, ReAgentLightningModule):
    def __init__(
        self,
        q_network,
        q_network_target,
        # Start SlateQTrainerParameters
        rl: rlp.RLParameters = field(  # noqa: B008
            default_factory=lambda: rlp.RLParameters(maxq_learning=False)
        ),
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        single_selection: bool = True,
        minibatch_size: int = 1024,
        evaluation: rlp.EvaluationParameters = field(  # noqa: B008
            default_factory=lambda: rlp.EvaluationParameters(calc_cpe_in_training=False)
        ),
    ) -> None:
        """
        Args:
            q_network: states, action -> q-value
            rl (optional): an instance of the RLParameter class, which
                defines relevant hyperparameters
            optimizer (optional): the optimizer class and
                optimizer hyperparameters for the q network(s) optimizer
            single_selection (optional): TBD
            minibatch_size (optional): the size of the minibatch
            evaluation (optional): TBD
        """
        super().__init__()
        self.rl_parameters = rl

        self.single_selection = single_selection

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = optimizer

    def configure_optimizers(self):
        optimizers = []

        optimizers.append(
            self.q_network_optimizer.make_optimizer(self.q_network.parameters())
        )

        target_params = list(self.q_network_target.parameters())
        source_params = list(self.q_network.parameters())
        optimizers.append(SoftUpdate(target_params, source_params, tau=self.tau))

        return optimizers

    def _action_docs(
        self,
        state: rlt.FeatureData,
        action: torch.Tensor,
        terminal_mask: Optional[torch.Tensor] = None,
    ) -> rlt.DocList:
        # for invalid indices, simply set action to 0 so we can batch index still
        if terminal_mask is not None:
            assert terminal_mask.shape == (
                action.shape[0],
            ), f"{terminal_mask.shape} != 0th dim of {action.shape}"
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

    def train_step_gen(self, training_batch: rlt.SlateQInput, batch_idx: int):
        assert isinstance(
            training_batch, rlt.SlateQInput
        ), f"learning input is a {type(training_batch)}"

        reward = training_batch.reward
        reward_mask = training_batch.reward_mask

        discount_tensor = torch.full_like(reward, self.gamma)

        if self.rl_parameters.maxq_learning:
            raise NotImplementedError("Q-Learning for SlateQ is not implemented")
        else:
            # SARSA (Use the target network)
            terminal_mask = (
                training_batch.not_terminal.to(torch.bool) == False
            ).squeeze(1)
            next_action_docs = self._action_docs(
                training_batch.next_state,
                training_batch.next_action,
                terminal_mask=terminal_mask,
            )
            value = next_action_docs.value
            if self.single_selection:
                value = F.softmax(value, dim=1)
            next_q_values = torch.sum(
                self._get_unmasked_q_values(
                    self.q_network_target,
                    training_batch.next_state,
                    next_action_docs,
                )
                * value,
                dim=1,
                keepdim=True,
            )

        # If not single selection, divide max-Q by N
        if not self.single_selection:
            _batch_size, slate_size = reward.shape
            next_q_values = next_q_values / slate_size

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
        self.log("td_loss", value_loss, prog_bar=True)
        yield result
