#!/usr/bin/env python3

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.training.ddpg_predictor import DDPGPredictor
from ml.rl.training.rl_trainer import DEFAULT_ADDITIONAL_FEATURE_TYPES


class DDPGTrainer(object):
    def __init__(
        self,
        parameters,
        state_normalization_parameters,
        action_normalization_parameters,
        use_gpu=False,
        additional_feature_types=DEFAULT_ADDITIONAL_FEATURE_TYPES,
        action_range=None,
    ) -> None:

        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
            self.dtype = torch.cuda.FloatTensor
        else:
            self.use_gpu = False
            self.dtype = torch.FloatTensor

        self._additional_feature_types = additional_feature_types
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.action_range = action_range
        self.state_dim = get_num_output_features(state_normalization_parameters)
        self.action_dim = get_num_output_features(action_normalization_parameters)

        # Shared params
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.use_seq_num_diff_as_time_diff = parameters.rl.use_seq_num_diff_as_time_diff
        self.final_layer_init = parameters.shared_training.final_layer_init
        if parameters.shared_training.optimizer == "ADAM":
            self.optimizer_func = torch.optim.Adam
        else:
            raise NotImplementedError(
                "{} optimizer not implemented".format(
                    parameters.shared_training.optimizer
                )
            )

        # Actor params
        self.actor_params = parameters.actor_training
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

        self.minibatch = 0
        self.reward_burnin = parameters.rl.reward_burnin

        if self.use_gpu:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

    def train(self, training_samples, evaluator=None, episode_values=None) -> None:
        states = Variable(torch.from_numpy(training_samples[0]).type(self.dtype))
        actions = Variable(torch.from_numpy(training_samples[1]).type(self.dtype))
        rewards = Variable(torch.from_numpy(training_samples[2]).type(self.dtype))
        next_states = Variable(torch.from_numpy(training_samples[3]).type(self.dtype))
        time_diffs = torch.tensor(training_samples[8]).type(self.dtype)
        discount_tensor = torch.tensor(
            np.array([self.gamma for x in range(len(rewards))])
        ).type(self.dtype)
        done = training_samples[5].astype(int)
        not_done_mask = Variable(torch.from_numpy(1 - done)).type(self.dtype)

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2
        q_s1_a1 = self.critic(torch.cat((states, actions), dim=1))
        next_actions = self.actor_target(next_states)

        next_state_actions = torch.cat((next_states, next_actions), dim=1)
        q_s2_a2 = self.critic_target(next_state_actions).detach().squeeze()
        filtered_q_s2_a2 = not_done_mask * q_s2_a2

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)

        self.minibatch += 1

        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_q_s2_a2)
        else:
            target_q_values = rewards
        # compute loss and update the critic network
        critic_predictions = q_s1_a1.squeeze()
        loss_critic = F.mse_loss(critic_predictions, target_q_values)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Optimize the actor network subject to the following:
        # max sum(Q(s1, a1)) or min -sum(Q(s1, a1))
        loss_actor = -self.critic(torch.cat((states, self.actor(states)), dim=1)).sum()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        if self.minibatch >= self.reward_burnin:
            # Use the soft update rule to update both target networks
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
        else:
            # Reward burnin: force target network
            self._soft_update(self.actor, self.actor_target, 1.0)
            self._soft_update(self.critic, self.critic_target, 1.0)

        if evaluator is not None:
            evaluator.report(
                loss_critic.cpu().data.numpy(),
                None,
                None,
                None,
                episode_values,
                None,
                None,
                critic_predictions.cpu().data.numpy(),
            )

    def _soft_update(self, network, target_network, tau) -> None:
        """ Target network update logic as defined in DDPG paper
        updated_params = tau * network_params + (1 - tau) * target_network_params
        :param network network with parameters to include in soft update
        :param target_network target network with params to soft update
        :param tau hyperparameter to control target tracking speed
        """
        for t_param, param in zip(target_network.parameters(), network.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)

    def internal_prediction(self, states, noisy=False) -> np.ndarray:
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        self.actor.eval()
        with torch.no_grad():
            state_examples = Variable(
                torch.from_numpy(np.array(states)).type(self.dtype)
            )
            actions = self.actor(state_examples)
        self.actor.train()
        actions = actions.cpu().data.numpy()

        if noisy:
            actions = [x + (self.noise.get_noise()) for x in actions]

        # Continuous action space
        if self.action_range:
            return np.array(
                [self.action_range * np.clip(action, -1, 1) for action in actions],
                dtype=np.float32,
            )
        # Discrete action space
        return np.array(actions, dtype=np.float32)

    def predictor(self, actor=True) -> DDPGPredictor:
        """Builds a DDPGPredictor.
        :param actor export the actor or the critic. If actor == True export
        the actor network, else export the critic network."""
        if actor:
            return DDPGPredictor.export_actor(
                self,
                self.state_normalization_parameters,
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
        if isinstance(state, np.ndarray):
            state = Variable(torch.from_numpy(state))

        x = state
        for i, activation in enumerate(self.activations):
            # TODO: (edoardoc) T30535967 Renable batchnorm when T30535876 is fixed
            # x = self.batch_norm_ops[i](x)
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
                # TODO: (edoardoc) T30535967 Renable batchnorm when T30535876 is fixed
                # x = self.batch_norm_ops[i](x)
                pass
            # Actions skip input layer
            elif i == 1:
                # TODO: (edoardoc) T30535967 Renable batchnorm when T30535876 is fixed
                # x = self.batch_norm_ops[i](x)
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

    def __init__(self, action_dim, theta=0.15, sigma=0.02, mu=0) -> None:
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.noise = np.zeros(self.action_dim, dtype=np.float32)

    def get_noise(self) -> np.ndarray:
        """dx = theta * (mu − prev_noise) + sigma * new_gaussian_noise"""
        term_1 = self.theta * (self.mu - self.noise)
        dx = term_1 + (self.sigma * np.random.randn(self.action_dim))
        return self.noise + dx

    def clear(self) -> None:
        self.noise = np.zeros(self.action_dim)
