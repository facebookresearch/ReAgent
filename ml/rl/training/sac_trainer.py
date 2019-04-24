#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.thrift.core.ttypes import SACModelParameters
from ml.rl.training.actor_predictor import ActorPredictor
from ml.rl.training.rl_exporter import ActorExporter, ParametricDQNExporter
from ml.rl.training.rl_trainer_pytorch import RLTrainer, rescale_torch_tensor


logger = logging.getLogger(__name__)


class SACTrainer(RLTrainer):
    """
    Soft Actor-Critic trainer as described in https://arxiv.org/pdf/1801.01290

    The actor is assumed to implement reparameterization trick.
    """

    def __init__(
        self,
        q1_network,
        value_network,
        value_network_target,
        actor_network,
        parameters: SACModelParameters,
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
        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1
        super().__init__(parameters, use_gpu=False)

        self.q1_network = q1_network
        self.q1_network_optimizer = self._get_optimizer(
            q1_network, parameters.training.q_network_optimizer
        )

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_optimizer = self._get_optimizer(
                q2_network, parameters.training.q_network_optimizer
            )

        self.value_network = value_network
        self.value_network_optimizer = self._get_optimizer(
            value_network, parameters.training.value_network_optimizer
        )
        self.value_network_target = value_network_target

        self.actor_network = actor_network
        self.actor_network_optimizer = self._get_optimizer(
            actor_network, parameters.training.actor_network_optimizer
        )

        self.entropy_temperature = parameters.training.entropy_temperature
        self.logged_action_uniform_prior = (
            parameters.training.logged_action_uniform_prior
        )

        # These ranges are only for Gym tests
        self.min_action_range_tensor_training = min_action_range_tensor_training
        self.max_action_range_tensor_training = max_action_range_tensor_training
        self.min_action_range_tensor_serving = min_action_range_tensor_serving
        self.max_action_range_tensor_serving = max_action_range_tensor_serving

    def warm_start_components(self):
        components = [
            "q1_network",
            "q1_network_optimizer",
            "value_network",
            "value_network_optimizer",
            "value_network_target",
            "actor_network",
            "actor_network_optimizer",
        ]
        if self.q2_network:
            components += ["q2_network", "q2_network_optimizer"]
        return components

    def train(self, training_batch) -> None:
        """
        IMPORTANT: the input action here is assumed to be preprocessed to match the
        range of the output of the actor.
        """
        if hasattr(training_batch, "as_parametric_sarsa_training_batch"):
            training_batch = training_batch.as_parametric_sarsa_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        state = learning_input.state
        action = learning_input.action
        reward = learning_input.reward
        discount = torch.full_like(reward, self.gamma)
        not_done_mask = learning_input.not_terminal

        if self._should_scale_action_in_train():
            action = rlt.FeatureVector(
                rescale_torch_tensor(
                    action.float_features,
                    new_min=self.min_action_range_tensor_training,
                    new_max=self.max_action_range_tensor_training,
                    prev_min=self.min_action_range_tensor_serving,
                    prev_max=self.max_action_range_tensor_serving,
                )
            )

        current_state_action = rlt.StateAction(state=state, action=action)

        q1_value = self.q1_network(current_state_action).q_value
        min_q_value = q1_value

        if self.q2_network:
            q2_value = self.q2_network(current_state_action).q_value
            min_q_value = torch.min(q1_value, q2_value)

        # Use the minimum as target, ensure no gradient going through
        min_q_value = min_q_value.detach()

        #
        # First, optimize value network; minimizing MSE between
        # V(s) & Q(s, a) - log(pi(a|s))
        #

        state_value = self.value_network(state.float_features)  # .q_value

        if self.logged_action_uniform_prior:
            log_prob_a = torch.zeros_like(min_q_value)
            target_value = min_q_value
        else:
            with torch.no_grad():
                log_prob_a = self.actor_network.get_log_prob(
                    state, action.float_features
                )
                log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                target_value = min_q_value - self.entropy_temperature * log_prob_a

        value_loss = F.mse_loss(state_value, target_value)
        value_loss.backward()
        self._maybe_run_optimizer(
            self.value_network_optimizer, self.minibatches_per_step
        )

        #
        # Second, optimize Q networks; minimizing MSE between
        # Q(s, a) & r + discount * V'(next_s)
        #

        with torch.no_grad():
            next_state_value = (
                self.value_network_target(learning_input.next_state.float_features)
                * not_done_mask.float()
            )

            target_q_value = reward + discount * next_state_value

        q1_loss = F.mse_loss(q1_value, target_q_value)
        q1_loss.backward()
        self._maybe_run_optimizer(self.q1_network_optimizer, self.minibatches_per_step)
        if self.q2_network:
            q2_loss = F.mse_loss(q2_value, target_q_value)
            q2_loss.backward()
            self._maybe_run_optimizer(
                self.q2_network_optimizer, self.minibatches_per_step
            )

        #
        # Lastly, optimize the actor; minimizing KL-divergence between action propensity
        # & softmax of value. Due to reparameterization trick, it ends up being
        # log_prob(actor_action) - Q(s, actor_action)
        #

        actor_output = self.actor_network(rlt.StateInput(state=state))

        state_actor_action = rlt.StateAction(
            state=state, action=rlt.FeatureVector(float_features=actor_output.action)
        )
        q1_actor_value = self.q1_network(state_actor_action).q_value
        min_q_actor_value = q1_actor_value
        if self.q2_network:
            q2_actor_value = self.q2_network(state_actor_action).q_value
            min_q_actor_value = torch.min(q1_actor_value, q2_actor_value)

        actor_loss = (
            self.entropy_temperature * actor_output.log_prob - min_q_actor_value
        )
        # Do this in 2 steps so we can log histogram of actor loss
        actor_loss_mean = actor_loss.mean()
        actor_loss_mean.backward()
        self._maybe_run_optimizer(
            self.actor_network_optimizer, self.minibatches_per_step
        )

        # Use the soft update rule to update both target networks
        self._maybe_soft_update(
            self.value_network,
            self.value_network_target,
            self.tau,
            self.minibatches_per_step,
        )

        # Logging at the end to schedule all the cuda operations first
        if (
            self.tensorboard_logging_freq is not None
            and self.minibatch % self.tensorboard_logging_freq == 0
        ):
            SummaryWriterContext.add_histogram("q1/logged_state_value", q1_value)
            if self.q2_network:
                SummaryWriterContext.add_histogram("q2/logged_state_value", q2_value)

            SummaryWriterContext.add_histogram("log_prob_a", log_prob_a)
            SummaryWriterContext.add_histogram("value_network/target", target_value)
            SummaryWriterContext.add_histogram(
                "q_network/next_state_value", next_state_value
            )
            SummaryWriterContext.add_histogram(
                "q_network/target_q_value", target_q_value
            )
            SummaryWriterContext.add_histogram(
                "actor/min_q_actor_value", min_q_actor_value
            )
            SummaryWriterContext.add_histogram(
                "actor/action_log_prob", actor_output.log_prob
            )
            SummaryWriterContext.add_histogram("actor/loss", actor_loss)

        self.loss_reporter.report(
            td_loss=float(q1_loss),
            reward_loss=None,
            logged_rewards=reward,
            model_values_on_logged_actions=q1_value,
            model_propensities=actor_output.log_prob.exp(),
            model_values=min_q_actor_value,
        )

    def _should_scale_action_in_train(self):
        if (
            self.min_action_range_tensor_training is not None
            and self.max_action_range_tensor_training is not None
            and self.min_action_range_tensor_serving is not None
            and self.max_action_range_tensor_serving is not None
        ):
            return True
        return False

    def internal_prediction(self, states):
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor_network.eval()
        state_examples = torch.from_numpy(np.array(states)).type(self.dtype)
        actions = self.actor_network(
            rlt.StateInput(rlt.FeatureVector(float_features=state_examples))
        )
        # clamp actions to make sure actions are in the range
        clamped_actions = torch.max(
            torch.min(actions.action, self.max_action_range_tensor_training),
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
