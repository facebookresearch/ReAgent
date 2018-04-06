#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ml.rl.training.ddpg_predictor import DDPGPredictor


class DDPGTrainer(object):
    def __init__(self, parameters, env_details) -> None:
        # Shared params
        self.env_details = env_details
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.learning_rate = parameters.shared_training.learning_rate
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        if parameters.shared_training.optimizer == 'ADAM':
            self.optimizer_func = torch.optim.Adam
        else:
            raise NotImplementedError("{} optimizer not implemented".format(
                parameters.shared_training.optimizer))

        # Actor params
        self.actor_params = parameters.actor_training
        self.actor_params.layers[0] = env_details.state_dim
        self.actor_params.layers[-1] = env_details.action_dim

        # Critic params
        self.critic_params = parameters.critic_training
        self.critic_params.layers[0] = env_details.state_dim + env_details.action_dim
        self.critic_params.layers[-1] = 1

    def train(self, predictor, training_samples) -> None:
        states = Variable(torch.from_numpy(training_samples[0]))
        actions = Variable(torch.from_numpy(training_samples[1]))
        rewards = Variable(torch.from_numpy(training_samples[2]))
        next_states = Variable(torch.from_numpy(training_samples[3]))

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2
        q_s1_a1 = predictor.critic(states, actions).squeeze()
        next_actions = predictor.actor_target(next_states).detach()
        q_s2_a2 = predictor.critic_target(
            next_states, next_actions.unsqueeze(dim=1)).squeeze().detach()
        target = rewards + (self.gamma * q_s2_a2)
        # compute loss and update the critic network
        loss_critic = F.mse_loss(q_s1_a1, target)
        predictor.critic_optimizer.zero_grad()
        loss_critic.backward()
        predictor.critic_optimizer.step()

        # Optimize the actor network subject to the following:
        # max sum(Q(s1, a1)) or min sum(-Q(s1, a1))
        a1 = predictor.actor(states).unsqueeze(dim=1)
        loss_actor = -1 * torch.sum(predictor.critic(states, a1))
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
