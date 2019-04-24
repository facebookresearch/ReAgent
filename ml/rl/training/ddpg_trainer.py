#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from copy import deepcopy
from typing import Dict

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl.models.base import ModelBase
from ml.rl.preprocessing.identify_types import CONTINUOUS
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.training.rl_trainer_pytorch import RLTrainer, rescale_torch_tensor


class DDPGTrainer(RLTrainer):
    def __init__(
        self,
        actor_network,
        critic_network,
        parameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        min_action_range_tensor_serving: torch.Tensor,
        max_action_range_tensor_serving: torch.Tensor,
        use_gpu: bool = False,
        use_all_avail_gpus: bool = False,
    ) -> None:

        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters

        for param in self.action_normalization_parameters.values():
            assert param.feature_type == CONTINUOUS, (
                "DDPG Actor features must be set to continuous (set to "
                + param.feature_type
                + ")"
            )

        self.state_dim = get_num_output_features(state_normalization_parameters)
        self.action_dim = min_action_range_tensor_serving.shape[1]
        self.num_features = self.state_dim + self.action_dim

        # Actor generates actions between -1 and 1 due to tanh output layer so
        # convert actions to range [-1, 1] before training.
        self.min_action_range_tensor_training = torch.ones(1, self.action_dim) * -1
        self.max_action_range_tensor_training = torch.ones(1, self.action_dim)
        self.min_action_range_tensor_serving = min_action_range_tensor_serving
        self.max_action_range_tensor_serving = max_action_range_tensor_serving

        # Shared params
        self.warm_start_model_path = parameters.shared_training.warm_start_model_path
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.minibatches_per_step = parameters.shared_training.minibatches_per_step or 1
        self.final_layer_init = parameters.shared_training.final_layer_init
        self._set_optimizer(parameters.shared_training.optimizer)

        # Actor params
        self.actor_params = parameters.actor_training
        self.actor = actor_network
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = self.optimizer_func(
            self.actor.parameters(),
            lr=self.actor_params.learning_rate,
            weight_decay=self.actor_params.l2_decay,
        )
        self.noise = OrnsteinUhlenbeckProcessNoise(self.action_dim)

        # Critic params
        self.critic_params = parameters.critic_training
        self.critic = self.q_network = critic_network
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = self.optimizer_func(
            self.critic.parameters(),
            lr=self.critic_params.learning_rate,
            weight_decay=self.critic_params.l2_decay,
        )

        # ensure state and action IDs have no intersection
        overlapping_features = set(state_normalization_parameters.keys()) & set(
            action_normalization_parameters.keys()
        )
        assert len(overlapping_features) == 0, (
            "There are some overlapping state and action features: "
            + str(overlapping_features)
        )

        RLTrainer.__init__(self, parameters, use_gpu, None)

        self.min_action_range_tensor_training = self.min_action_range_tensor_training.type(
            self.dtype
        )
        self.max_action_range_tensor_training = self.max_action_range_tensor_training.type(
            self.dtype
        )
        self.min_action_range_tensor_serving = self.min_action_range_tensor_serving.type(
            self.dtype
        )
        self.max_action_range_tensor_serving = self.max_action_range_tensor_serving.type(
            self.dtype
        )

    def train(self, training_batch: rlt.TrainingBatch) -> None:
        if hasattr(training_batch, "as_parametric_sarsa_training_batch"):
            training_batch = training_batch.as_parametric_sarsa_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        state = learning_input.state

        # As far as ddpg is concerned all actions are [-1, 1] due to actor tanh
        action = rlt.FeatureVector(
            rescale_torch_tensor(
                learning_input.action.float_features,
                new_min=self.min_action_range_tensor_training,
                new_max=self.max_action_range_tensor_training,
                prev_min=self.min_action_range_tensor_serving,
                prev_max=self.max_action_range_tensor_serving,
            )
        )

        rewards = learning_input.reward
        next_state = learning_input.next_state
        time_diffs = learning_input.time_diff
        discount_tensor = torch.full_like(rewards, self.gamma)
        not_done_mask = learning_input.not_terminal

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2
        q_s1_a1 = self.critic.forward(
            rlt.StateAction(state=state, action=action)
        ).q_value
        next_action = rlt.FeatureVector(
            float_features=self.actor_target(
                rlt.StateAction(state=next_state, action=None)
            ).action
        )

        q_s2_a2 = self.critic_target.forward(
            rlt.StateAction(state=next_state, action=next_action)
        ).q_value
        filtered_q_s2_a2 = not_done_mask.float() * q_s2_a2

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)

        target_q_values = rewards + (discount_tensor * filtered_q_s2_a2)

        # compute loss and update the critic network
        critic_predictions = q_s1_a1
        loss_critic = self.q_network_loss(critic_predictions, target_q_values.detach())
        loss_critic_for_eval = loss_critic.detach()
        loss_critic.backward()
        self._maybe_run_optimizer(self.critic_optimizer, self.minibatches_per_step)

        # Optimize the actor network subject to the following:
        # max mean(Q(s1, a1)) or min -mean(Q(s1, a1))
        actor_output = self.actor(rlt.StateAction(state=state, action=None))
        loss_actor = -(
            self.critic.forward(
                rlt.StateAction(
                    state=state,
                    action=rlt.FeatureVector(float_features=actor_output.action),
                )
            ).q_value.mean()
        )

        # Zero out both the actor and critic gradients because we need
        #   to backprop through the critic to get to the actor
        loss_actor.backward()
        self._maybe_run_optimizer(self.actor_optimizer, self.minibatches_per_step)

        # Use the soft update rule to update both target networks
        self._maybe_soft_update(
            self.actor, self.actor_target, self.tau, self.minibatches_per_step
        )
        self._maybe_soft_update(
            self.critic, self.critic_target, self.tau, self.minibatches_per_step
        )

        self.loss_reporter.report(
            td_loss=float(loss_critic_for_eval),
            reward_loss=None,
            model_values_on_logged_actions=critic_predictions,
        )

    def internal_prediction(self, states, noisy=False) -> np.ndarray:
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor.eval()
        # TODO: Handle states being sequences
        state_examples = rlt.FeatureVector(
            float_features=torch.from_numpy(np.array(states)).type(self.dtype)
        )
        action = self.actor(rlt.StateAction(state=state_examples, action=None)).action

        self.actor.train()

        action = rescale_torch_tensor(
            action,
            new_min=self.min_action_range_tensor_serving,
            new_max=self.max_action_range_tensor_serving,
            prev_min=self.min_action_range_tensor_training,
            prev_max=self.max_action_range_tensor_training,
        )

        action = action.cpu().data.numpy()
        if noisy:
            action = [x + (self.noise.get_noise()) for x in action]

        return np.array(action, dtype=np.float32)


class ActorNet(nn.Module):
    def __init__(self, layers, activations, fl_init) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations

        assert len(layers) >= 2, "Invalid layer schema {} for actor network".format(
            layers
        )

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            # If last layer use simple uniform init (as outlined in DDPG paper)
            if i + 1 == len(layers[1:]):
                init.uniform_(self.layers[i].weight, -fl_init, fl_init)
                init.uniform_(self.layers[i].bias, -fl_init, fl_init)
            # Else use fan in uniform init (as outlined in DDPG paper)
            else:
                fan_in_init(self.layers[i].weight, self.layers[i].bias)

    def forward(self, state) -> torch.FloatTensor:
        """ Forward pass for actor network. Assumes activation names are
        valid pytorch activation names.
        :param state state as list of state features
        """
        x = state
        for i, activation in enumerate(self.activations):
            x = self.batch_norm_ops[i](x)
            x = self.layers[i](x)
            if activation == "linear":
                continue
            elif activation == "tanh":
                activation_func = torch.tanh
            else:
                activation_func = getattr(F, activation)
            x = activation_func(x)
        return x


class ActorNetModel(ModelBase):
    def __init__(
        self,
        layers,
        activations,
        fl_init,
        state_dim,
        action_dim,
        use_gpu,
        use_all_avail_gpus,
        dnn=None,
    ) -> None:
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Allow dnn to be explicitly given (when exporting to cpu)
        if dnn is None:
            self.dnn = ActorNet(layers=layers, activations=activations, fl_init=fl_init)
        else:
            self.dnn = dnn

        # `network` might be a nn.DataParallel when running on multiple devices
        self.network = self.dnn
        if use_gpu:
            self.dnn.cuda()
            if use_all_avail_gpus:
                self.network = nn.DataParallel(self.dnn)

    def input_prototype(self) -> rlt.StateInput:
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def cpu_model(self):
        cpu_dnn = deepcopy(self.dnn)
        cpu_dnn.cpu()

        return self.__class__(
            layers=self.dnn.layers,
            activations=self.dnn.activations,
            fl_init=None,  # Not saved anyway, and we're providing the initialized dnn
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_gpu=False,
            use_all_avail_gpus=False,
            dnn=cpu_dnn,
        )

    def forward(self, input: rlt.StateInput) -> rlt.ActorOutput:
        """ Forward pass for actor network. Assumes activation names are
        valid pytorch activation names.
        :param input StateInput containing float_features
        """
        if input.state.float_features is None:
            raise NotImplementedError("Not implemented for non-float_features!")

        action = self.network.forward(state=input.state.float_features)
        return rlt.ActorOutput(action=action)


class CriticNet(nn.Module):
    def __init__(self, layers, activations, fl_init, action_dim) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations

        assert len(layers) >= 3, "Invalid layer schema {} for critic network".format(
            layers
        )

        assert layers[-1] == 1, "Only one output node for the critic net"

        for i, layer in enumerate(layers[1:]):
            # Batch norm only applied to pre-action layers
            if i == 0:
                self.layers.append(nn.Linear(layers[i], layer))
                self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            elif i == 1:
                self.layers.append(nn.Linear(layers[i] + action_dim, layer))
                self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            # Actions skip input layer
            else:
                self.layers.append(nn.Linear(layers[i], layer))

            # If last layer use simple uniform init (as outlined in DDPG paper)
            if i + 1 == len(layers[1:]):
                init.uniform_(self.layers[i].weight, -fl_init, fl_init)
                init.uniform_(self.layers[i].bias, -fl_init, fl_init)
            # Else use fan in uniform init (as outlined in DDPG paper)
            else:
                fan_in_init(self.layers[i].weight, self.layers[i].bias)

    def forward(self, state_action) -> torch.FloatTensor:
        """ Forward pass for critic network. Assumes activation names are
        valid pytorch activation names.
        :param state_action tensor of state & actions concatted
        """
        if isinstance(state_action, list):
            return self.forward_split(*state_action)

        state_dim = self.layers[0].in_features
        state = state_action[:, :state_dim]
        action = state_action[:, state_dim:]
        return self.forward_split(state, action)

    def forward_split(self, state, action) -> torch.FloatTensor:
        x = state
        for i, activation in enumerate(self.activations):
            if i == 0:
                x = self.batch_norm_ops[i](x)
            # Actions skip input layer
            elif i == 1:
                x = self.batch_norm_ops[i](x)
                x = torch.cat((x, action), dim=1)

            x = self.layers[i](x)
            if activation == "linear":
                continue
            elif activation == "tanh":
                activation_func = torch.tanh
            else:
                activation_func = getattr(F, activation)
            x = activation_func(x)
        return x


class CriticNetModel(ModelBase):
    def __init__(
        self,
        layers,
        activations,
        fl_init,
        state_dim,
        action_dim,
        use_gpu,
        use_all_avail_gpus,
        dnn=None,
    ) -> None:
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Allow dnn to be explicitly given (for exporting as cpu)
        if dnn is None:
            self.dnn = CriticNet(
                layers=layers,
                activations=activations,
                fl_init=fl_init,
                action_dim=action_dim,
            )
        else:
            self.dnn = dnn

        # `network` might be a nn.DataParallel if on multiple devices
        self.network = self.dnn
        if use_gpu:
            self.dnn.cuda()
            if use_all_avail_gpus:
                self.network = nn.DataParallel(self.dnn)

    def cpu_model(self):
        cpu_dnn = deepcopy(self.dnn)
        cpu_dnn.cpu()

        return self.__class__(
            layers=self.dnn.layers,
            activations=self.dnn.activations,
            fl_init=None,  # Not saved anyway, and we're providing the initialized dnn
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_gpu=False,
            use_all_avail_gpus=False,
            dnn=cpu_dnn,
        )

    def forward(self, input: rlt.StateAction) -> rlt.SingleQValue:
        """ Forward pass for critic network. Assumes activation names are
        valid pytorch activation names.
        :param input ml.rl.types.StateAction of combined states and actions
        """
        return rlt.SingleQValue(
            q_value=self.network.forward(
                [input.state.float_features, input.action.float_features]
            )
        )

    def input_prototype(self) -> rlt.StateAction:
        return rlt.StateAction(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim)),
            action=rlt.FeatureVector(float_features=torch.randn(1, self.action_dim)),
        )


def fan_in_init(weight_tensor, bias_tensor) -> None:
    """ Fan in initialization as described in DDPG paper."""
    val_range = 1.0 / np.sqrt(weight_tensor.size(1))
    init.uniform_(weight_tensor, -val_range, val_range)
    init.uniform_(bias_tensor, -val_range, val_range)


class OrnsteinUhlenbeckProcessNoise:
    """ Exploration noise process with temporally correlated noise. Used to
    explore in physical environments w/momentum. Outlined in DDPG paper."""

    def __init__(self, action_dim, theta=0.15, sigma=0.2, mu=0) -> None:
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.noise = np.zeros(self.action_dim, dtype=np.float32)

    def get_noise(self) -> np.ndarray:
        """dx = theta * (mu − prev_noise) + sigma * new_gaussian_noise"""
        term_1 = self.theta * (self.mu - self.noise)
        dx = term_1 + (self.sigma * np.random.randn(self.action_dim))
        self.noise = self.noise + dx
        return self.noise

    def clear(self) -> None:
        self.noise = np.zeros(self.action_dim)
