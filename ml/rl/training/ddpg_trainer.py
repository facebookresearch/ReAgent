#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ml.rl.training.ddpg_predictor import DDPGPredictor


class DDPGTrainer(object):
    def __init__(self, parameters, env_details) -> None:
        # Shared params
        self.env_details = env_details
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.final_layer_init = parameters.shared_training.final_layer_init
        if parameters.shared_training.optimizer == 'ADAM':
            self.optimizer_func = torch.optim.Adam
        else:
            raise NotImplementedError("{} optimizer not implemented".format(
                parameters.shared_training.optimizer))

        # Actor params
        self.actor_params = parameters.actor_training
        self.actor_params.layers[0] = env_details.state_dim
        self.actor_params.layers[-1] = env_details.action_dim
        self.noise_generator = OrnsteinUhlenbeckProcessNoise(env_details.action_dim)

        # Critic params
        self.critic_params = parameters.critic_training
        self.critic_params.layers[0] = env_details.state_dim
        self.critic_params.layers[-1] = 1

    def train(self, predictor, training_samples) -> None:
        states = Variable(torch.from_numpy(training_samples[0]))
        actions = Variable(torch.from_numpy(training_samples[1]))
        rewards = Variable(torch.from_numpy(training_samples[2]))
        next_states = Variable(torch.from_numpy(training_samples[3]))
        done = training_samples[5].astype(int)
        not_done_mask = Variable(torch.from_numpy(1 - done)).type(torch.FloatTensor)

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2
        q_s1_a1 = predictor.critic(states, actions)
        next_actions = predictor.actor_target(next_states)
        q_s2_a2 = predictor.critic_target(next_states, next_actions).detach().squeeze()
        filtered_q_s2_a2 = not_done_mask * q_s2_a2
        target_q_values = rewards + (self.gamma * filtered_q_s2_a2)
        # compute loss and update the critic network
        loss_critic = F.mse_loss(q_s1_a1.squeeze(), target_q_values)
        predictor.critic_optimizer.zero_grad()
        loss_critic.backward()
        predictor.critic_optimizer.step()

        # Optimize the actor network subject to the following:
        # max sum(Q(s1, a1)) or min -sum(Q(s1, a1))
        loss_actor = -predictor.critic(states, predictor.actor(states)).sum()
        predictor.actor_optimizer.zero_grad()
        loss_actor.backward()
        predictor.actor_optimizer.step()

        # Use the soft update rule to update both target networks
        self._soft_update(predictor.actor, predictor.actor_target, self.tau)
        self._soft_update(predictor.critic, predictor.critic_target, self.tau)

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

    def predictor(self) -> DDPGPredictor:
        """Builds a DDPGPredictor."""
        return DDPGPredictor.export_actor(self)


class OrnsteinUhlenbeckProcessNoise():
    """ Exploration noise process with temporally correlated noise. Used to
    explore in physical environments w/momentum. Outlined in DDPG paper."""
    def __init__(self, action_dim, theta=0.15, sigma=0.20, mu=0) -> None:
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
