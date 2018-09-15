#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import AdditionalFeatureTypes
from ml.rl.training.ddpg_trainer import (
    ActorNet,
    CriticNet,
    OrnsteinUhlenbeckProcessNoise,
)
from ml.rl.training.knn_dqn_predictor import KNNDQNPredictor
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage
from torch.autograd import Variable


class KNNDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool = False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
    ) -> None:

        self.state_normalization_parameters = state_normalization_parameters

        self.state_dim = get_num_output_features(state_normalization_parameters)
        self.num_actions = parameters.num_actions
        self.action_dim = parameters.action_dim
        self.k = parameters.k

        # Shared params
        self.warm_start_model_path = parameters.shared_training.warm_start_model_path
        self.minibatch_size = parameters.shared_training.minibatch_size
        self.final_layer_init = parameters.shared_training.final_layer_init
        self._set_optimizer(parameters.shared_training.optimizer)

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

        super(KNNDQNTrainer, self).__init__(
            parameters, use_gpu, additional_feature_types
        )

        self.action_embedding = nn.Embedding(
            self.num_actions, self.action_dim, sparse=True  # max_norm=1.0
        )

        if self.use_gpu:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
            self.action_embedding.cuda()

    def train(
        self, training_samples: TrainingDataPage, evaluator=None, episode_values=None
    ) -> None:

        self.minibatch += 1
        states = torch.from_numpy(training_samples.states).type(self.dtype)
        actions = torch.from_numpy(training_samples.actions).type(self.dtypelong)
        # As far as ddpg is concerned all actions are [-1, 1] due to actor tanh

        rewards = torch.from_numpy(training_samples.rewards).type(self.dtype)
        next_states = torch.from_numpy(training_samples.next_states).type(self.dtype)
        time_diffs = torch.tensor(training_samples.time_diffs).type(self.dtype)
        discount_tensor = torch.tensor(np.full(len(rewards), self.gamma)).type(
            self.dtype
        )
        not_done_mask = Variable(
            torch.from_numpy(training_samples.not_terminals.astype(int))
        ).type(self.dtype)

        # Optimize the critic network subject to mean squared error:
        # L = ([r + gamma * Q(s2, a2)] - Q(s1, a1)) ^ 2

        action_embeddings = self.action_embedding(actions.view(-1))
        q_s1_a1 = self.critic(torch.cat((states, action_embeddings), dim=1))

        next_actions = self.actor_target(next_states)
        # TODO: We might replace this with a KNN lookup to speed up
        inner_products = F.linear(next_actions, self.action_embedding.weight)
        _top_k_similarity, top_k_idx = inner_products.topk(self.k)
        k_nearest_next_actions = self.action_embedding(top_k_idx.view(-1))
        k_next_states = next_states.repeat(1, self.k).view(-1, self.state_dim)
        next_state_actions = torch.cat((k_next_states, k_nearest_next_actions), dim=1)
        q_s2_a2 = self.critic_target(next_state_actions).squeeze()
        not_done_mask = (
            not_done_mask.view(-1, 1).expand(-1, self.k).contiguous().view(-1)
        )
        filtered_q_s2_a2 = not_done_mask * q_s2_a2

        # compute loss and update the critic network
        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)
        discount_tensor = (
            discount_tensor.view(-1, 1).expand(-1, self.k).contiguous().view(-1)
        )

        rewards = rewards.view(-1, 1).expand(-1, self.k).contiguous().view(-1)
        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_q_s2_a2)
        else:
            target_q_values = rewards

        target_q_values, max_q_idx = (
            target_q_values.view(-1, self.k).contiguous().max(dim=1)
        )

        critic_predictions = q_s1_a1.squeeze()
        loss_critic = F.mse_loss(critic_predictions, target_q_values.detach())
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # TODO: the paper was unclear on what should be the gradient for actor
        # given that the nearest neighbor is not differentiable. Here, we make
        # the simplest thing by just plugging output of the actor directly in
        # the critic so it's differentiable.

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
                None,
                critic_predictions.cpu().data.numpy(),
                None,
            )

    def actor_predictor(self) -> KNNDQNPredictor:
        return KNNDQNPredictor.export_actor(
            self,
            self.state_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )

    def critic_predictor(self) -> KNNDQNPredictor:
        return KNNDQNPredictor.export_critic(
            self,
            self.state_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )

    def predictor(self, action_names=None) -> KNNDQNPredictor:
        return KNNDQNPredictor.export(
            self,
            self.state_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
            action_names=action_names,
        )
