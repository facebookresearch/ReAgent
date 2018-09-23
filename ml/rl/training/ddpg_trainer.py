#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.preprocessing.preprocessor_net import sort_features_by_normalization
from ml.rl.thrift.core.ttypes import AdditionalFeatureTypes
from ml.rl.training.ddpg_predictor import DDPGPredictor
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    RLTrainer,
    rescale_torch_tensor,
)
from ml.rl.training.training_data_page import TrainingDataPage
from torch.autograd import Variable


class DDPGTrainer(RLTrainer):
    def __init__(
        self,
        parameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        min_action_range_tensor_serving: torch.tensor,
        max_action_range_tensor_serving: torch.tensor,
        use_gpu: bool = False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
    ) -> None:

        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters

        self.state_dim = get_num_output_features(state_normalization_parameters)
        self.action_dim = min_action_range_tensor_serving.shape[1]

        # Actor generates actions between -1 and 1 due to tanh output layer so
        # convert actions to range [-1, 1] before training.
        self.min_action_range_tensor_training = torch.ones(1, self.action_dim) * -1
        self.max_action_range_tensor_training = torch.ones(1, self.action_dim)
        self.min_action_range_tensor_serving = min_action_range_tensor_serving
        self.max_action_range_tensor_serving = max_action_range_tensor_serving

        # Shared params
        self.warm_start_model_path = parameters.shared_training.warm_start_model_path
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.final_layer_init = parameters.shared_training.final_layer_init
        self._set_optimizer(parameters.shared_training.optimizer)

        # Actor params
        self.actor_params = parameters.actor_training
        assert (
            self.actor_params.activations[-1] == "tanh"
        ), "Actor final layer activation must be tanh"
        self.actor_params.layers[0] = self.state_dim
        self.actor_params.layers[-1] = self.action_dim
        self.noise_generator = OrnsteinUhlenbeckProcessNoise(self.action_dim)
        self.actor = ActorNet(
            self.actor_params.layers,
            self.actor_params.activations,
            self.final_layer_init,
        )
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = self.optimizer_func(
            self.actor.parameters(), lr=self.actor_params.learning_rate
        )
        self.noise = self.noise_generator

        # Critic params
        self.critic_params = parameters.critic_training
        self.critic_params.layers[0] = self.state_dim
        self.critic_params.layers[-1] = 1
        self.critic = CriticNet(
            self.critic_params.layers,
            self.critic_params.activations,
            self.final_layer_init,
            self.action_dim,
        )
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = self.optimizer_func(
            self.critic.parameters(),
            lr=self.critic_params.learning_rate,
            weight_decay=self.critic_params.l2_decay,
        )

        RLTrainer.__init__(self, parameters, use_gpu, additional_feature_types, None)

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

        if self.use_gpu:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

    def train(
        self, training_samples: TrainingDataPage, evaluator=None, episode_values=None
    ) -> None:
        if self.minibatch == 0:
            # Assume that the tensors are the right shape after the first minibatch
            assert training_samples.states.shape[0] == self.minibatch_size, (
                "Invalid shape: " + str(training_samples.states.shape)
            )
            assert training_samples.actions.shape[0] == self.minibatch_size, (
                "Invalid shape: " + str(training_samples.actions.shape)
            )
            assert training_samples.rewards.shape == torch.Size(
                [self.minibatch_size, 1]
            ), "Invalid shape: " + str(training_samples.rewards.shape)
            assert (
                training_samples.episode_values is None
                or training_samples.episode_values.shape
                == training_samples.rewards.shape
            ), (
                "Invalid shape: " + str(training_samples.episode_values.shape)
            )
            assert (
                training_samples.next_states.shape == training_samples.states.shape
            ), (
                "Invalid shape: " + str(training_samples.next_states.shape)
            )
            assert (
                training_samples.not_terminals.shape == training_samples.rewards.shape
            ), (
                "Invalid shape: " + str(training_samples.not_terminals.shape)
            )
            if self.use_seq_num_diff_as_time_diff:
                assert (
                    training_samples.time_diffs.shape == training_samples.rewards.shape
                ), (
                    "Invalid shape: " + str(training_samples.time_diffs.shape)
                )

        self.minibatch += 1
        states = training_samples.states.detach().requires_grad_(True)
        actions = training_samples.actions.detach().requires_grad_(True)

        # As far as ddpg is concerned all actions are [-1, 1] due to actor tanh
        actions = rescale_torch_tensor(
            actions,
            new_min=self.min_action_range_tensor_training,
            new_max=self.max_action_range_tensor_training,
            prev_min=self.min_action_range_tensor_serving,
            prev_max=self.max_action_range_tensor_serving,
        )
        rewards = training_samples.rewards
        next_states = torch.tensor(training_samples.next_states, requires_grad=True)
        time_diffs = training_samples.time_diffs
        discount_tensor = torch.tensor(np.full(rewards.shape, self.gamma)).type(
            self.dtype
        )
        not_done_mask = training_samples.not_terminals

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2
        q_s1_a1 = self.critic(torch.cat((states, actions), dim=1))
        next_actions = self.actor_target(next_states)

        next_state_actions = torch.cat((next_states, next_actions), dim=1)
        q_s2_a2 = self.critic_target(next_state_actions)
        filtered_q_s2_a2 = not_done_mask * q_s2_a2

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)

        if self.use_reward_burnin and self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_q_s2_a2)

        # compute loss and update the critic network
        critic_predictions = q_s1_a1
        loss_critic = self.q_network_loss(critic_predictions, target_q_values.detach())
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Optimize the actor network subject to the following:
        # max sum(Q(s1, a1)) or min -sum(Q(s1, a1))
        loss_actor = -self.critic(torch.cat((states, self.actor(states)), dim=1)).sum()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        if self.use_reward_burnin and self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.actor, self.actor_target, 1.0)
            self._soft_update(self.critic, self.critic_target, 1.0)
        else:
            # Use the soft update rule to update both target networks
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)

        if evaluator is not None:
            evaluator.report(
                loss_critic.cpu().data.numpy(),
                None,
                None,
                None,
                episode_values,
                None,
                None,
                None,
                critic_predictions.cpu().data.numpy(),
                None,
            )

    def internal_prediction(self, states, noisy=False) -> np.ndarray:
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor.eval()
        state_examples = torch.from_numpy(np.array(states)).type(self.dtype)
        actions = self.actor(state_examples)

        self.actor.train()

        actions = rescale_torch_tensor(
            actions,
            new_min=self.min_action_range_tensor_serving,
            new_max=self.max_action_range_tensor_serving,
            prev_min=self.min_action_range_tensor_training,
            prev_max=self.max_action_range_tensor_training,
        )

        actions = actions.cpu().data.numpy()
        if noisy:
            actions = [x + (self.noise.get_noise()) for x in actions]

        return np.array(actions, dtype=np.float32)

    def predictor(self, actor=True) -> DDPGPredictor:
        """Builds a DDPGPredictor.
        :param actor export the actor or the critic. If actor == True export
        the actor network, else export the critic network."""
        if actor:
            return DDPGPredictor.export_actor(
                self,
                self.state_normalization_parameters,
                self.min_action_range_tensor_serving,
                self.max_action_range_tensor_serving,
                self._additional_feature_types.int_features,
                self.use_gpu,
            )
        return DDPGPredictor.export_critic(
            self,
            self.state_normalization_parameters,
            self.action_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )


class ActorNet(nn.Module):
    def __init__(self, layers, activations, fl_init) -> None:
        super(ActorNet, self).__init__()
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
                fan_in_init(self.layers[i].weight)

    def forward(self, state) -> torch.FloatTensor:
        """ Forward pass for actor network. Assumes activation names are
        valid pytorch activation names.
        :param state state as list of state features
        """
        x = state
        for i, activation in enumerate(self.activations):
            x = self.batch_norm_ops[i](x)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, layers, activations, fl_init, action_dim) -> None:
        super(CriticNet, self).__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations

        assert len(layers) >= 3, "Invalid layer schema {} for critic network".format(
            layers
        )

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
                fan_in_init(self.layers[i].weight)

    def forward(self, state_action) -> torch.FloatTensor:
        """ Forward pass for critic network. Assumes activation names are
        valid pytorch activation names.
        :param state_action tensor of state & actions concatted
        """
        state_dim = self.layers[0].in_features
        state = state_action[:, :state_dim]
        action = state_action[:, state_dim:]

        x = state
        for i, activation in enumerate(self.activations):
            if i == 0:
                x = self.batch_norm_ops[i](x)
            # Actions skip input layer
            elif i == 1:
                x = self.batch_norm_ops[i](x)
                x = torch.cat((x, action), dim=1)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))
        return x


def fan_in_init(tensor) -> None:
    """ Fan in initialization as described in DDPG paper."""
    val_range = 1. / np.sqrt(tensor.size(1))
    init.uniform_(tensor, -val_range, val_range)


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


def construct_action_scale_tensor(action_norm_params, action_scale_overrides):
    """Construct tensors that will rescale each action value on each dimension i
    from [min_serving_value[i], max_serving_value[i]] to [-1, 1] for training.
    """
    sorted_features, _ = sort_features_by_normalization(action_norm_params)
    min_action_array = np.zeros((1, len(sorted_features)))
    max_action_array = np.zeros((1, len(sorted_features)))

    for idx, feature_id in enumerate(sorted_features):
        if feature_id in action_scale_overrides:
            min_action_array[0][idx] = action_scale_overrides[feature_id][0]
            max_action_array[0][idx] = action_scale_overrides[feature_id][1]
        else:
            min_action_array[0][idx] = action_norm_params[feature_id].min_value
            max_action_array[0][idx] = action_norm_params[feature_id].max_value

    min_action_range_tensor_serving = torch.from_numpy(min_action_array)
    max_action_range_tensor_serving = torch.from_numpy(max_action_array)
    return min_action_range_tensor_serving, max_action_range_tensor_serving
