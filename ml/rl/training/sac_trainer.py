#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.thrift.core.ttypes import SACModelParameters
from ml.rl.torch_utils import rescale_torch_tensor
from ml.rl.training.rl_trainer_pytorch import RLTrainer


logger = logging.getLogger(__name__)


class SACTrainer(RLTrainer):
    """
    Soft Actor-Critic trainer as described in https://arxiv.org/pdf/1801.01290

    The actor is assumed to implement reparameterization trick.
    """

    def __init__(
        self,
        q1_network,
        actor_network,
        parameters: SACModelParameters,
        use_gpu=False,
        value_network=None,
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
        super().__init__(parameters, use_gpu=use_gpu)

        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1
        assert self.minibatches_per_step == 1, (
            "minibatches_per_step must be 1. Gradient accumation doesn't work "
            "with actor-critic"
        )

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
        if self.value_network is not None:
            self.value_network_optimizer = self._get_optimizer(
                value_network, parameters.training.value_network_optimizer
            )
            self.value_network_target = copy.deepcopy(self.value_network)
        else:
            self.q1_network_target = copy.deepcopy(self.q1_network)
            self.q2_network_target = copy.deepcopy(self.q2_network)

        self.actor_network = actor_network
        self.actor_network_optimizer = self._get_optimizer(
            actor_network, parameters.training.actor_network_optimizer
        )

        self.alpha_optimizer = None
        device = "cuda" if use_gpu else "cpu"
        if parameters.training.alpha_optimizer is not None:
            if parameters.training.target_entropy is not None:
                self.target_entropy = parameters.training.target_entropy
            elif min_action_range_tensor_training is not None:
                self.target_entropy = -np.prod(
                    min_action_range_tensor_training.cpu().data.numpy().shape
                ).item()
            else:
                self.target_entropy = -1

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = self._get_optimizer_func(
                parameters.training.alpha_optimizer.optimizer
            )(
                [self.log_alpha],
                lr=parameters.training.alpha_optimizer.learning_rate,
                weight_decay=parameters.training.alpha_optimizer.l2_decay,
            )

        self.entropy_temperature = (
            parameters.training.entropy_temperature
            if parameters.training.entropy_temperature is not None
            else 0.1
        )
        self.logged_action_uniform_prior = (
            parameters.training.logged_action_uniform_prior
        )

        self.add_kld_to_loss = bool(parameters.training.action_embedding_kld_weight)

        if self.add_kld_to_loss:
            self.kld_weight = parameters.training.action_embedding_kld_weight
            self.action_emb_mean = torch.tensor(
                parameters.training.action_embedding_mean, device=device
            )
            self.action_emb_variance = torch.tensor(
                parameters.training.action_embedding_variance, device=device
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
            "actor_network",
            "actor_network_optimizer",
        ]
        if self.q2_network:
            components += ["q2_network", "q2_network_optimizer"]
        if self.value_network:
            components += [
                "value_network",
                "value_network_optimizer",
                "value_network_target",
            ]
        else:
            components += ["q1_network_target"]
            if self.q2_network:
                components += ["q2_network_target"]
        return components

    @torch.no_grad()  # type: ignore
    def train(self, training_batch) -> None:
        """
        IMPORTANT: the input action here is assumed to be preprocessed to match the
        range of the output of the actor.
        """
        if hasattr(training_batch, "as_policy_network_training_batch"):
            training_batch = training_batch.as_policy_network_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        state = learning_input.state
        action = learning_input.action
        reward = learning_input.reward
        discount = torch.full_like(reward, self.gamma)
        not_done_mask = learning_input.not_terminal

        if self._should_scale_action_in_train():
            action = action._replace(
                float_features=rescale_torch_tensor(
                    action.float_features,
                    new_min=self.min_action_range_tensor_training,
                    new_max=self.max_action_range_tensor_training,
                    prev_min=self.min_action_range_tensor_serving,
                    prev_max=self.max_action_range_tensor_serving,
                )
            )

        # We need to zero out grad here because gradient from actor update
        # should not be used in Q-network update
        self.actor_network_optimizer.zero_grad()
        self.q1_network_optimizer.zero_grad()
        if self.q2_network is not None:
            self.q2_network_optimizer.zero_grad()
        if self.value_network is not None:
            self.value_network_optimizer.zero_grad()

        with torch.enable_grad():
            #
            # First, optimize Q networks; minimizing MSE between
            # Q(s, a) & r + discount * V'(next_s)
            #

            current_state_action = rlt.PreprocessedStateAction(
                state=state, action=action
            )
            q1_value = self.q1_network(current_state_action).q_value
            if self.q2_network:
                q2_value = self.q2_network(current_state_action).q_value
            actor_output = self.actor_network(rlt.PreprocessedState(state=state))

            # Optimize Alpha
            if self.alpha_optimizer is not None:
                alpha_loss = -(
                    self.log_alpha
                    * (actor_output.log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.entropy_temperature = self.log_alpha.exp()

            with torch.no_grad():
                if self.value_network is not None:
                    next_state_value = self.value_network_target(
                        learning_input.next_state.float_features
                    )
                else:
                    next_state_actor_output = self.actor_network(
                        rlt.PreprocessedState(state=learning_input.next_state)
                    )
                    next_state_actor_action = rlt.PreprocessedStateAction(
                        state=learning_input.next_state,
                        action=rlt.PreprocessedFeatureVector(
                            float_features=next_state_actor_output.action
                        ),
                    )
                    next_state_value = self.q1_network_target(
                        next_state_actor_action
                    ).q_value

                    if self.q2_network is not None:
                        target_q2_value = self.q2_network_target(
                            next_state_actor_action
                        ).q_value
                        next_state_value = torch.min(next_state_value, target_q2_value)

                    log_prob_a = self.actor_network.get_log_prob(
                        learning_input.next_state, next_state_actor_output.action
                    )
                    log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                    next_state_value -= self.entropy_temperature * log_prob_a

                if self.gamma > 0.0:
                    target_q_value = (
                        reward + discount * next_state_value * not_done_mask.float()
                    )
                else:
                    # This is useful in debugging instability issues
                    target_q_value = reward

            q1_loss = F.mse_loss(q1_value, target_q_value)
            q1_loss.backward()
            self._maybe_run_optimizer(
                self.q1_network_optimizer, self.minibatches_per_step
            )
            if self.q2_network:
                q2_loss = F.mse_loss(q2_value, target_q_value)
                q2_loss.backward()
                self._maybe_run_optimizer(
                    self.q2_network_optimizer, self.minibatches_per_step
                )

            #
            # Second, optimize the actor; minimizing KL-divergence between action propensity
            # & softmax of value. Due to reparameterization trick, it ends up being
            # log_prob(actor_action) - Q(s, actor_action)
            #

            state_actor_action = rlt.PreprocessedStateAction(
                state=state,
                action=rlt.PreprocessedFeatureVector(
                    float_features=actor_output.action
                ),
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

            if self.add_kld_to_loss:
                action_batch_m = torch.mean(actor_output.action, axis=0)
                action_batch_v = torch.var(actor_output.action, axis=0)
                kld = (
                    0.5
                    * (
                        (action_batch_v + (action_batch_m - self.action_emb_mean) ** 2)
                        / self.action_emb_variance
                        - 1
                        + self.action_emb_variance.log()
                        - action_batch_v.log()
                    ).sum()
                )

                actor_loss_mean += self.kld_weight * kld

            actor_loss_mean.backward()
            self._maybe_run_optimizer(
                self.actor_network_optimizer, self.minibatches_per_step
            )

            #
            # Lastly, if applicable, optimize value network; minimizing MSE between
            # V(s) & E_a~pi(s) [ Q(s,a) - log(pi(a|s)) ]
            #

            if self.value_network is not None:
                state_value = self.value_network(state.float_features)

                if self.logged_action_uniform_prior:
                    log_prob_a = torch.zeros_like(min_q_actor_value)
                    target_value = min_q_actor_value
                else:
                    with torch.no_grad():
                        log_prob_a = actor_output.log_prob
                        log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                        target_value = (
                            min_q_actor_value - self.entropy_temperature * log_prob_a
                        )

                value_loss = F.mse_loss(state_value, target_value.detach())
                value_loss.backward()
                self._maybe_run_optimizer(
                    self.value_network_optimizer, self.minibatches_per_step
                )

        # Use the soft update rule to update the target networks
        if self.value_network is not None:
            self._maybe_soft_update(
                self.value_network,
                self.value_network_target,
                self.tau,
                self.minibatches_per_step,
            )
        else:
            self._maybe_soft_update(
                self.q1_network,
                self.q1_network_target,
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
            self.tensorboard_logging_freq is not None
            and self.minibatch % self.tensorboard_logging_freq == 0
        ):
            SummaryWriterContext.add_histogram("q1/logged_state_value", q1_value)
            if self.q2_network:
                SummaryWriterContext.add_histogram("q2/logged_state_value", q2_value)

            SummaryWriterContext.add_histogram("log_prob_a", log_prob_a)
            if self.value_network:
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
            if self.add_kld_to_loss:
                SummaryWriterContext.add_histogram("kld/mean", action_batch_m)
                SummaryWriterContext.add_histogram("kld/var", action_batch_v)
                SummaryWriterContext.add_scalar("kld/kld", kld)

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

    def internal_prediction(self, states, test=False):
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(rlt.PreprocessedState.from_tensor(states))
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
