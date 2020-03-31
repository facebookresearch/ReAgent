#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import List

import ml.rl.parameters as rlp
import ml.rl.types as rlt
import torch
from ml.rl.training.dqn_trainer_base import DQNTrainerBase


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
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.learning_rate,
            # weight_decay=parameters.training.l2_decay,
        )

    def warm_start_components(self) -> List[str]:
        components = ["q_network", "q_network_target", "q_network_optimizer"]
        return components

    @torch.no_grad()  # type: ignore
    def get_detached_q_values_target(
        self,
        tiled_state: rlt.PreprocessedTiledFeatureVector,
        action: rlt.PreprocessedSlateFeatureVector,
    ) -> torch.Tensor:
        """ Gets the q values from the target network """
        return self.get_slate_q_value(self.q_network_target, tiled_state, action)

    def get_slate_q_value(
        self,
        q_network,
        tiled_state: rlt.PreprocessedTiledFeatureVector,
        action: rlt.PreprocessedSlateFeatureVector,
    ) -> torch.Tensor:
        """ Gets the q values from the model and target networks """
        input = rlt.PreprocessedStateAction(
            state=tiled_state.as_preprocessed_feature_vector(),
            action=action.as_preprocessed_feature_vector(),
        )
        q_value = self.q_network_target(input).q_value
        q_value = (
            q_value.view(action.float_features.shape[0], action.float_features.shape[1])
            * action.item_mask
            * action.item_probability
        )
        return q_value.sum(dim=1, keepdim=True)

    @torch.no_grad()  # type: ignore
    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        learning_input = training_batch.training_input
        assert isinstance(
            learning_input, rlt.PreprocessedSlateQInput
        ), f"learning input is a {type(learning_input)}"
        self.minibatch += 1

        reward = learning_input.reward
        reward_mask = learning_input.reward_mask
        not_done_mask = learning_input.not_terminal

        discount_tensor = torch.full_like(reward, self.gamma)

        if self.maxq_learning:
            raise NotImplementedError("Q-Learning for SlateQ is not implemented")
        else:
            # SARSA (Use the target network)
            next_q_values = self.get_detached_q_values_target(
                learning_input.tiled_next_state, learning_input.next_action
            )

        filtered_max_q_vals = next_q_values * not_done_mask.float()

        target_q_values = reward + (discount_tensor * filtered_max_q_vals)
        target_q_values = target_q_values[reward_mask]

        with torch.enable_grad():
            # Get Q-value of action taken
            current_state_action = rlt.PreprocessedStateAction(
                state=learning_input.tiled_state.as_preprocessed_feature_vector(),
                action=learning_input.action.as_preprocessed_feature_vector(),
            )
            q_values = self.q_network(current_state_action).q_value.view(
                *reward_mask.shape
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
