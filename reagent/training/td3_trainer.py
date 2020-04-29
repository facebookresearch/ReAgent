#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.parameters import TD3ModelParameters
from reagent.tensorboardX import SummaryWriterContext
from reagent.torch_utils import rescale_torch_tensor
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class TD3Trainer(RLTrainer):
    """
    Twin Delayed Deep Deterministic Policy Gradient algorithm trainer
    as described in https://arxiv.org/pdf/1802.09477
    """

    def __init__(
        self,
        q1_network,
        actor_network,
        parameters: TD3ModelParameters,
        use_gpu=False,
        q2_network=None,
        min_action_range_tensor_training=None,
        max_action_range_tensor_training=None,
        min_action_range_tensor_serving=None,
        max_action_range_tensor_serving=None,
    ) -> None:
        """
        Args:
            The four args below are provided for integration with other
            environments (e.g., Gym):
            min_action_range_tensor_training / max_action_range_tensor_training:
                min / max value of actions at training time
            min_action_range_tensor_serving / max_action_range_tensor_serving:
                min / max value of actions at serving time
        """
        super().__init__(parameters.rl, use_gpu=use_gpu)

        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1

        self.q1_network = q1_network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q1_network_optimizer = self._get_optimizer(
            q1_network, parameters.training.q_network_optimizer
        )

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_target = copy.deepcopy(self.q2_network)
            self.q2_network_optimizer = self._get_optimizer(
                q2_network, parameters.training.q_network_optimizer
            )

        self.actor_network = actor_network
        self.actor_network_target = copy.deepcopy(self.actor_network)
        self.actor_network_optimizer = self._get_optimizer(
            actor_network, parameters.training.actor_network_optimizer
        )

        self.exploration_noise = parameters.training.exploration_noise
        self.initial_exploration_ts = parameters.training.initial_exploration_ts
        self.target_policy_smoothing = parameters.training.target_policy_smoothing
        self.noise_clip = parameters.training.noise_clip
        self.delayed_policy_update = parameters.training.delayed_policy_update

        # These ranges are only for Gym tests
        self.min_action_range_tensor_training = min_action_range_tensor_training
        self.max_action_range_tensor_training = max_action_range_tensor_training
        self.min_action_range_tensor_serving = min_action_range_tensor_serving
        self.max_action_range_tensor_serving = max_action_range_tensor_serving

    def warm_start_components(self):
        components = [
            "q1_network",
            "q1_network_target",
            "q1_network_optimizer",
            "actor_network",
            "actor_network_target",
            "actor_network_optimizer",
        ]
        if self.q2_network:
            components += ["q2_network", "q2_network_target", "q2_network_optimizer"]

        return components

    def train(self, training_batch: rlt.PreprocessedPolicyNetworkInput) -> None:
        """
        IMPORTANT: the input action here is assumed to be preprocessed to match the
        range of the output of the actor.
        """
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_policy_network_training_batch()

        assert isinstance(training_batch, rlt.PreprocessedPolicyNetworkInput)

        self.minibatch += 1

        state = training_batch.state
        action = training_batch.action
        next_state = training_batch.next_state
        reward = training_batch.reward
        not_done_mask = training_batch.not_terminal

        action = self._maybe_scale_action_in_train(action.float_features)

        max_action = (
            self.max_action_range_tensor_training
            if self.max_action_range_tensor_training
            else torch.ones(action.shape, device=self.device)
        )
        min_action = (
            self.min_action_range_tensor_serving
            if self.min_action_range_tensor_serving
            else -torch.ones(action.shape, device=self.device)
        )

        # Compute current value estimates
        current_state_action = rlt.PreprocessedStateAction(
            state=state, action=rlt.PreprocessedFeatureVector(float_features=action)
        )
        q1_value = self.q1_network(current_state_action).q_value
        if self.q2_network:
            q2_value = self.q2_network(current_state_action).q_value
        actor_action = self.actor_network(rlt.PreprocessedState(state=state)).action

        # Generate target = r + y * min (Q1(s',pi(s')), Q2(s',pi(s')))
        with torch.no_grad():
            next_actor = self.actor_network_target(
                rlt.PreprocessedState(state=next_state)
            ).action
            next_actor += (
                torch.randn_like(next_actor) * self.target_policy_smoothing
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actor = torch.max(torch.min(next_actor, max_action), min_action)
            next_state_actor = rlt.PreprocessedStateAction(
                state=next_state,
                action=rlt.PreprocessedFeatureVector(float_features=next_actor),
            )
            next_state_value = self.q1_network_target(next_state_actor).q_value

            if self.q2_network is not None:
                next_state_value = torch.min(
                    next_state_value, self.q2_network_target(next_state_actor).q_value
                )

            target_q_value = (
                reward + self.gamma * next_state_value * not_done_mask.float()
            )

        # Optimize Q1 and Q2
        q1_loss = F.mse_loss(q1_value, target_q_value)
        q1_loss.backward()
        self._maybe_run_optimizer(self.q1_network_optimizer, self.minibatches_per_step)
        if self.q2_network:
            q2_loss = F.mse_loss(q2_value, target_q_value)
            q2_loss.backward()
            self._maybe_run_optimizer(
                self.q2_network_optimizer, self.minibatches_per_step
            )

        # Only update actor and target networks after a fixed number of Q updates
        if self.minibatch % self.delayed_policy_update == 0:
            actor_loss = -self.q1_network(
                rlt.PreprocessedStateAction(
                    state=state,
                    action=rlt.PreprocessedFeatureVector(float_features=actor_action),
                )
            ).q_value.mean()
            actor_loss.backward()
            self._maybe_run_optimizer(
                self.actor_network_optimizer, self.minibatches_per_step
            )

            # Use the soft update rule to update the target networks
            self._maybe_soft_update(
                self.q1_network,
                self.q1_network_target,
                self.tau,
                self.minibatches_per_step,
            )
            self._maybe_soft_update(
                self.actor_network,
                self.actor_network_target,
                self.tau,
                self.minibatches_per_step,
            )
            if self.q2_network is not None:
                self._maybe_soft_update(
                    self.q2_network,
                    self.q2_network_target,
                    self.tau,
                    self.minibatches_per_step,
                )

        # Logging at the end to schedule all the cuda operations first
        if (
            self.tensorboard_logging_freq != 0
            and self.minibatch % self.tensorboard_logging_freq == 0
        ):
            SummaryWriterContext.add_histogram("q1/logged_state_value", q1_value)
            if self.q2_network:
                SummaryWriterContext.add_histogram("q2/logged_state_value", q2_value)

            SummaryWriterContext.add_histogram(
                "q_network/next_state_value", next_state_value
            )
            SummaryWriterContext.add_histogram(
                "q_network/target_q_value", target_q_value
            )
            SummaryWriterContext.add_histogram("actor/loss", actor_loss)

        self.loss_reporter.report(
            td_loss=float(q1_loss),
            reward_loss=None,
            logged_rewards=reward,
            model_values_on_logged_actions=q1_value,
        )

    def _maybe_scale_action_in_train(self, action):
        if (
            self.min_action_range_tensor_training is not None
            and self.max_action_range_tensor_training is not None
            and self.min_action_range_tensor_serving is not None
            and self.max_action_range_tensor_serving is not None
        ):
            action = rescale_torch_tensor(
                action,
                new_min=self.min_action_range_tensor_training,
                new_max=self.max_action_range_tensor_training,
                prev_min=self.min_action_range_tensor_serving,
                prev_max=self.max_action_range_tensor_serving,
            )
        return action

    def internal_prediction(self, states, test=False):
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(
                rlt.PreprocessedState.from_tensor(states)
            ).action

        if not test:
            if self.minibatch < self.initial_exploration_ts:
                actions = (
                    torch.rand_like(actions)
                    * (
                        self.max_action_range_tensor_training
                        - self.min_action_range_tensor_training
                    )
                    + self.min_action_range_tensor_training
                )
            else:
                actions += torch.randn_like(actions) * self.exploration_noise

        # clamp actions to make sure actions are in the range
        clamped_actions = torch.max(
            torch.min(actions, self.max_action_range_tensor_training),
            self.min_action_range_tensor_training,
        )
        rescaled_actions = rescale_torch_tensor(
            clamped_actions,
            new_min=self.min_action_range_tensor_serving,
            new_max=self.max_action_range_tensor_serving,
            prev_min=self.min_action_range_tensor_training,
            prev_max=self.max_action_range_tensor_training,
        )

        self.actor_network.train()
        return rescaled_actions
