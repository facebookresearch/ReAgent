#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List

import reagent.parameters as rlp
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.training.dqn_trainer_base import DQNTrainerBase


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlateQTrainerParameters:
    rl: rlp.RLParameters = field(
        default_factory=lambda: rlp.RLParameters(maxq_learning=False)
    )
    optimizer: str = "ADAM"
    learning_rate: float = 0.001
    minibatch_size: int = 1024
    evaluation: rlp.EvaluationParameters = field(
        default_factory=lambda: rlp.EvaluationParameters(calc_cpe_in_training=False)
    )


class SlateQTrainer(DQNTrainerBase):
    def __init__(
        self,
        q_network,
        q_network_target,
        parameters: SlateQTrainerParameters,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(parameters.rl, use_gpu=use_gpu)
        self.minibatches_per_step = 1
        self.minibatch_size = parameters.minibatch_size

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(parameters.optimizer)
        # pyre-fixme[16]: `SlateQTrainer` has no attribute `optimizer_func`.
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.learning_rate,
            # weight_decay=parameters.training.l2_decay,
        )

    def warm_start_components(self) -> List[str]:
        components = ["q_network", "q_network_target", "q_network_optimizer"]
        return components

    def get_slate_q_value(
        self, q_network, state: rlt.FeatureData, action: rlt.SlateFeatureVector
    ) -> torch.Tensor:
        """ Gets the q values from the model and target networks """
        return (
            self._get_unmask_q_values(q_network, state, action)
            * action.item_mask
            * action.item_probability
        ).sum(dim=1, keepdim=True)

    def _get_unmask_q_values(
        self, q_network, state: rlt.FeatureData, action: rlt.SlateFeatureVector
    ) -> torch.Tensor:
        batch_size, slate_size, _ = action.float_features.shape
        return q_network(
            state.repeat_interleave(slate_size, dim=0),
            action.as_preprocessed_feature_vector(),
        ).view(batch_size, slate_size)

    @torch.no_grad()
    def train(self, training_batch: rlt.SlateQInput):
        assert isinstance(
            training_batch, rlt.SlateQInput
        ), f"learning input is a {type(training_batch)}"
        self.minibatch += 1

        reward = training_batch.reward
        reward_mask = training_batch.reward_mask

        discount_tensor = torch.full_like(reward, self.gamma)

        if self.maxq_learning:
            raise NotImplementedError("Q-Learning for SlateQ is not implemented")
        else:
            # SARSA (Use the target network)
            next_q_values = self.get_slate_q_value(
                self.q_network_target,
                training_batch.next_state,
                training_batch.next_action,
            )

        filtered_max_q_vals = next_q_values * training_batch.not_terminal.float()

        target_q_values = reward + (discount_tensor * filtered_max_q_vals)
        target_q_values = target_q_values[reward_mask]

        with torch.enable_grad():
            # Get Q-value of action taken
            q_values = self._get_unmask_q_values(
                self.q_network, training_batch.state, training_batch.action
            )[reward_mask]
            all_action_scores = q_values.detach()

            value_loss = self.q_network_loss(q_values, target_q_values)
            td_loss = value_loss.detach()
            value_loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        self.loss_reporter.report(
            td_loss=td_loss, model_values_on_logged_actions=all_action_scores
        )
