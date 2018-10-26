#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import ml.rl.types as rlt
import torch
import torch.nn.functional as F
from ml.rl.thrift.core.ttypes import SACModelParameters
from ml.rl.training._parametric_dqn_predictor import _ParametricDQNPredictor
from ml.rl.training.actor_predictor import ActorPredictor
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.rl_exporter import ParametricDQNExporter
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
        value_network,
        value_network_target,
        actor_network,
        parameters: SACModelParameters,
        q2_network=None,
    ) -> None:
        super(SACTrainer, self).__init__(
            parameters,
            use_gpu=False,
            additional_feature_types=None,
            gradient_handler=None,
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
        self.value_network_optimizer = self._get_optimizer(
            value_network, parameters.training.value_network_optimizer
        )
        self.value_network_target = value_network_target

        self.actor_network = actor_network
        self.actor_network_optimizer = self._get_optimizer(
            actor_network, parameters.training.actor_network_optimizer
        )

        self.entropy_temperature = parameters.training.entropy_temperature

    def train(self, training_batch, evaluator=None) -> None:
        if hasattr(training_batch, "as_parametric_sarsa_training_batch"):
            training_batch = training_batch.as_parametric_sarsa_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        s = learning_input.state
        a = learning_input.action.float_features
        reward = learning_input.reward
        discount = torch.full_like(reward, self.gamma)
        not_done_mask = learning_input.not_terminal

        current_state_action = rlt.StateAction(
            state=learning_input.state, action=learning_input.action
        )

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

        state_value = self.value_network(s.float_features)  # .q_value

        with torch.no_grad():
            log_prob_a = self.actor_network.get_log_prob(s, a)
            target_value = min_q_value - self.entropy_temperature * log_prob_a

        value_loss = F.mse_loss(state_value, target_value)
        self.value_network_optimizer.zero_grad()
        value_loss.backward()
        self.value_network_optimizer.step()

        #
        # Second, optimize Q networks; minimizing MSE between
        # Q(s, a) & r + discount * V'(next_s)
        #

        with torch.no_grad():
            next_state_value = (
                self.value_network_target(learning_input.next_state.float_features)
                * not_done_mask
            )

            if self.minibatch < self.reward_burnin:
                target_q_value = reward
            else:
                target_q_value = reward + discount * next_state_value

        q1_loss = F.mse_loss(q1_value, target_q_value)
        self.q1_network_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_network_optimizer.step()
        if self.q2_network:
            q2_loss = F.mse_loss(q2_value, target_q_value)
            self.q2_network_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_network_optimizer.step()

        #
        # Lastly, optimize the actor; minimizing KL-divergence between action propensity
        # & softmax of value. Due to reparameterization trick, it ends up being
        # log_prob(actor_action) - Q(s, actor_action)
        #

        actor_output = self.actor_network(rlt.StateInput(state=learning_input.state))

        state_actor_action = rlt.StateAction(
            state=s, action=rlt.FeatureVector(float_features=actor_output.action)
        )
        q1_actor_value = self.q1_network(state_actor_action).q_value
        min_q_actor_value = q1_actor_value
        if self.q2_network:
            q2_actor_value = self.q2_network(state_actor_action).q_value
            min_q_actor_value = torch.min(q1_actor_value, q2_actor_value)

        actor_loss = torch.mean(
            self.entropy_temperature * actor_output.log_prob - min_q_actor_value
        )
        self.actor_network_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.value_network, self.value_network_target, 1.0)
        else:
            # Use the soft update rule to update both target networks
            self._soft_update(self.value_network, self.value_network_target, self.tau)

        if evaluator is not None:
            # FIXME
            self.evaluate(evaluator)

    def evaluate(self, evaluator: Evaluator):
        # FIXME
        evaluator.report(
            self.loss.cpu().numpy(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.all_action_scores.cpu().numpy(),
            None,
        )

    def actor_predictor(
        self, feature_extractor=None, output_trasnformer=None, net_container=None
    ):
        actor_network = self.actor_network.cpu_model()
        if net_container is not None:
            actor_network = net_container(actor_network)
        predictor = ActorPredictor.export(
            actor_network, feature_extractor, output_trasnformer
        )
        self.actor_network.train()
        return predictor

    def critic_predictor(
        self, feature_extractor=None, output_trasnformer=None, net_container=None
    ) -> _ParametricDQNPredictor:
        # TODO: We should combine the two Q functions
        q_network = self.q1_network.cpu_model()
        if net_container is not None:
            q_network = net_container(q_network)
        predictor = ParametricDQNExporter(
            q_network, feature_extractor, output_trasnformer
        ).export()
        self.q1_network.train()
        return predictor
