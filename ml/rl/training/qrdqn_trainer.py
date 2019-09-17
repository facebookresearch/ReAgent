#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import ml.rl.types as rlt
import torch
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class QRDQNTrainer(DQNTrainerBase):
    """
    Implementation of QR-DQN (Quantile Regression Deep Q-Nework)

    See https://arxiv.org/abs/1710.10044 for details
    """

    def __init__(
        self,
        q_network,
        q_network_target,
        parameters: DiscreteActionModelParameters,
        use_gpu=False,
        metrics_to_score=None,
        reward_network=None,
        q_network_cpe=None,
        q_network_cpe_target=None,
    ) -> None:
        super().__init__(
            parameters,
            use_gpu=use_gpu,
            metrics_to_score=metrics_to_score,
            actions=parameters.actions,
        )

        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1
        self._actions = parameters.actions if parameters.actions is not None else []

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.rainbow.c51_l2_decay,
        )

        self.num_atoms = parameters.rainbow.num_atoms
        self.quantiles = (
            (0.5 + torch.arange(self.num_atoms, device=self.device).float())
            / float(self.num_atoms)
        ).view(1, -1)

        self._initialize_cpe(
            parameters, reward_network, q_network_cpe, q_network_cpe_target
        )

        self.reward_boosts = torch.zeros([1, len(self._actions)], device=self.device)
        if parameters.rl.reward_boost is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

    def warm_start_components(self):
        components = ["q_network", "q_network_target", "q_network_optimizer"]
        if self.reward_network is not None:
            components += [
                "reward_network",
                "reward_network_optimizer",
                "q_network_cpe",
                "q_network_cpe_target",
                "q_network_cpe_optimizer",
            ]
        return components

    @torch.no_grad()  # type: ignore
    def train(self, training_batch):
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_discrete_maxq_training_batch()

        learning_input = training_batch.training_input
        state = rlt.PreprocessedState(state=learning_input.state)
        next_state = rlt.PreprocessedState(state=learning_input.next_state)
        rewards = self.boost_rewards(learning_input.reward, learning_input.action)
        discount_tensor = torch.full_like(rewards, self.gamma)
        possible_next_actions_mask = learning_input.possible_next_actions_mask.float()
        possible_actions_mask = learning_input.possible_actions_mask.float()

        self.minibatch += 1
        not_done_mask = learning_input.not_terminal.float()

        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, learning_input.time_diff.float())
        if self.multi_steps is not None:
            discount_tensor = torch.pow(self.gamma, learning_input.step.float())

        next_qf = self.q_network_target.dist(next_state)

        if self.maxq_learning:
            # Select distribution corresponding to max valued action
            next_q_values = (
                self.q_network(next_state)
                if self.double_q_learning
                else next_qf.mean(2)
            )
            next_action = self.argmax_with_mask(
                next_q_values, possible_next_actions_mask
            )
            next_qf = next_qf[range(rewards.shape[0]), next_action.reshape(-1)]
        else:
            next_qf = (next_qf * learning_input.next_action.unsqueeze(-1)).sum(1)

        # Build target distribution
        target_Q = rewards + discount_tensor * not_done_mask * next_qf

        with torch.enable_grad():
            current_qf = self.q_network.dist(state)

            # for reporting only
            all_q_values = current_qf.mean(2).detach()

            current_qf = (current_qf * learning_input.action.unsqueeze(-1)).sum(1)

            # (batch, atoms) -> (atoms, batch, 1) -> (atoms, batch, atoms)
            td = target_Q.t().unsqueeze(-1) - current_qf
            loss = (
                self.huber(td) * (self.quantiles - (td.detach() < 0).float()).abs()
            ).mean()

            loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        # Get Q-values of next states, used in computing cpe
        all_next_action_scores = self.q_network(next_state).q_values.detach()

        logged_action_idxs = learning_input.action.argmax(dim=1, keepdim=True)
        reward_loss, model_rewards, model_propensities = self._calculate_cpes(
            training_batch,
            state,
            next_state,
            all_q_values,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )

        model_action_idxs = self.argmax_with_mask(
            all_q_values,
            possible_actions_mask if self.maxq_learning else learning_input.action,
        )
        self.loss_reporter.report(
            td_loss=loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            model_values=all_q_values,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=model_action_idxs,
        )

    @torch.no_grad()  # type: ignore
    def internal_prediction(self, input):
        """
        Only used by Gym
        """
        self.q_network.eval()
        q_values = self.q_network(rlt.PreprocessedState.from_tensor(input))
        q_values = q_values.q_values.cpu()
        self.q_network.train()

        return q_values

    @torch.no_grad()  # type: ignore
    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts,  # type: ignore
            dim=1,
            keepdim=True,
        )
        return rewards + reward_boosts

    def argmax_with_mask(self, q_values, possible_actions_mask):
        # Set q-values of impossible actions to a very large negative number.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values = q_values + self.ACTION_NOT_POSSIBLE_VAL * (1 - possible_actions_mask)
        return q_values.argmax(1)

    # Used to prevent warning when a.shape != b.shape
    def huber(self, x):
        return torch.where(x.abs() < 1, 0.5 * x.pow(2), x.abs() - 0.5)

    @torch.no_grad()  # type: ignore
    def get_detached_q_values(self, state):
        """ Gets the q values from the model and target networks """
        input = rlt.PreprocessedState(state=state)
        q_values = self.q_network(input).q_values
        q_values_target = self.q_network_target(input).q_values
        return q_values, q_values_target
