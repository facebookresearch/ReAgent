#!/usr/bin/env python3

import logging
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from ml.rl.models.convolutional_network import ConvolutionalNetwork
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import (
    AdditionalFeatureTypes,
    DiscreteActionModelParameters,
)
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class DQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool = False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
        gradient_handler=None,
        use_all_avail_gpus: bool = False,
    ) -> None:

        self.double_q_learning = parameters.rainbow.double_q_learning
        self.warm_start_model_path = parameters.training.warm_start_model_path
        self.minibatch_size = parameters.training.minibatch_size
        self._actions = parameters.actions if parameters.actions is not None else []

        if parameters.training.cnn_parameters is None:
            self.state_normalization_parameters: Optional[
                Dict[int, NormalizationParameters]
            ] = state_normalization_parameters
            self.num_features = get_num_output_features(state_normalization_parameters)
            logger.info("Number of state features: " + str(self.num_features))
            parameters.training.layers[0] = self.num_features
        else:
            self.state_normalization_parameters = None
        parameters.training.layers[-1] = self.num_actions

        RLTrainer.__init__(
            self, parameters, use_gpu, additional_feature_types, gradient_handler
        )

        self.reward_boosts = torch.zeros([1, len(self._actions)]).type(self.dtype)
        if parameters.rl.reward_boost is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

        if parameters.rainbow.dueling_architecture:
            self.q_network = DuelingQNetwork(
                parameters.training.layers, parameters.training.activations
            )
        else:
            if parameters.training.cnn_parameters is None:
                self.q_network = FullyConnectedNetwork(
                    parameters.training.layers, parameters.training.activations
                )
            else:
                self.q_network = ConvolutionalNetwork(
                    parameters.training.cnn_parameters,
                    parameters.training.layers,
                    parameters.training.activations,
                )

        self.q_network_target = deepcopy(self.q_network)
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(), lr=parameters.training.learning_rate
        )

        if parameters.training.cnn_parameters is None:
            self.reward_network = FullyConnectedNetwork(
                parameters.training.layers, parameters.training.activations
            )
        else:
            self.reward_network = ConvolutionalNetwork(
                parameters.training.cnn_parameters,
                parameters.training.layers,
                parameters.training.activations,
            )
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()
            self.reward_network.cuda()

            if use_all_avail_gpus:
                self.q_network = torch.nn.DataParallel(self.q_network)
                self.q_network_target = torch.nn.DataParallel(self.q_network_target)
                self.reward_network = torch.nn.DataParallel(self.reward_network)

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def calculate_q_values(self, states):
        return self.q_network(states).detach()

    def get_max_q_values(self, states, possible_actions, double_q_learning):
        """
        Used in Q-learning update.
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :param double_q_learning: bool to use double q-learning
        """
        if double_q_learning:
            q_values = self.q_network(states).detach()
            q_values_target = self.q_network_target(states).detach()
            # Set q-values of impossible actions to a very large negative number.
            inverse_pna = 1 - possible_actions
            impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
            q_values += impossible_action_penalty
            # Select max_q action after scoring with online network
            max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
            # Use q_values from target network for max_q action from online q_network
            # to decouble selection & scoring, preventing overestimation of q-values
            q_values = torch.gather(q_values_target, 1, max_indicies)
            return q_values
        else:
            q_values = self.q_network_target(states).detach()
            # Set q-values of impossible actions to a very large negative number.
            inverse_pna = 1 - possible_actions
            impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
            q_values += impossible_action_penalty
            return torch.max(q_values, dim=1, keepdim=True)[0]

    def get_next_action_q_values(self, states, next_actions):
        """
        Used in SARSA update.
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param next_actions: Numpy array with shape (batch_size, action_dim).
        """
        q_values = self.q_network_target(states).detach()
        return torch.sum(q_values * next_actions, dim=1, keepdim=True)

    def train(
        self, training_samples: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        if self.minibatch == 0:
            # Assume that the tensors are the right shape after the first minibatch
            assert training_samples.states.shape[0] == self.minibatch_size, (
                "Invalid shape: " + str(training_samples.states.shape)
            )
            assert training_samples.actions.shape == torch.Size(
                [self.minibatch_size, len(self._actions)]
            ), "Invalid shape: " + str(training_samples.actions.shape)
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
            if training_samples.possible_next_actions is not None:
                assert (
                    training_samples.possible_next_actions.shape
                    == training_samples.actions.shape
                ), (
                    "Invalid shape: "
                    + str(training_samples.possible_next_actions.shape)
                )
            if training_samples.propensities is not None:
                assert (
                    training_samples.propensities.shape
                    == training_samples.rewards.shape
                ), (
                    "Invalid shape: " + str(training_samples.propensities.shape)
                )

        # Apply reward boost if specified
        reward_boosts = torch.sum(
            training_samples.actions.float() * self.reward_boosts, dim=1, keepdim=True
        )
        boosted_rewards = training_samples.rewards + reward_boosts

        self.minibatch += 1
        states = training_samples.states.detach().requires_grad_(True)
        actions = training_samples.actions
        rewards = boosted_rewards
        next_states = training_samples.next_states
        discount_tensor = torch.full(
            training_samples.time_diffs.shape, self.gamma
        ).type(self.dtype)
        not_done_mask = training_samples.not_terminals

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(training_samples.time_diffs)

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            possible_next_actions = training_samples.possible_next_actions
            next_q_values = self.get_max_q_values(
                next_states, possible_next_actions, self.double_q_learning
            )
        else:
            # SARSA
            next_actions = training_samples.next_actions
            next_q_values = self.get_next_action_q_values(next_states, next_actions)

        filtered_next_q_vals = next_q_values * not_done_mask

        if self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        # Get Q-value of action taken
        all_q_values = self.q_network(states)
        self.all_action_scores = all_q_values.detach()
        q_values = torch.sum(all_q_values * actions, 1, keepdim=True)

        loss = self.q_network_loss(q_values, target_q_values)
        self.loss = loss.detach()

        self.q_network_optimizer.zero_grad()
        loss.backward()
        if self.gradient_handler:
            self.gradient_handler(self.q_network.parameters())
        self.q_network_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        # get reward estimates
        reward_estimates = self.reward_network(states)
        self.reward_estimates = reward_estimates.detach()
        reward_estimates_for_logged_actions = reward_estimates.gather(
            1, actions.argmax(dim=1, keepdim=True)
        )
        reward_loss = F.mse_loss(reward_estimates_for_logged_actions, rewards)
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        if evaluator is not None:
            self.evaluate(
                evaluator,
                training_samples.actions,
                training_samples.propensities,
                boosted_rewards,
                training_samples.episode_values,
            )

    def evaluate(
        self,
        evaluator: Evaluator,
        logged_actions: torch.Tensor,
        logged_propensities: Optional[torch.Tensor],
        logged_rewards: torch.Tensor,
        logged_values: Optional[torch.Tensor],
    ):
        self.model_propensities, model_values_on_logged_actions, maxq_action_idxs = (
            None,
            None,
            None,
        )
        if self.all_action_scores is not None:
            self.model_propensities = Evaluator.softmax(
                self.all_action_scores.cpu().numpy(), self.rl_temperature
            )
            maxq_action_idxs = (
                self.all_action_scores.argmax(dim=1, keepdim=True).cpu().numpy()
            )
            if logged_actions is not None:
                model_values_on_logged_actions = (
                    torch.sum(
                        (logged_actions * self.all_action_scores), dim=1, keepdim=True
                    )
                    .cpu()
                    .numpy()
                )

        evaluator.report(
            self.loss.cpu().numpy(),
            logged_actions.cpu().numpy(),
            logged_propensities.cpu().numpy()
            if logged_propensities is not None
            else None,
            logged_rewards.cpu().numpy(),
            logged_values.cpu().numpy() if logged_values is not None else None,
            self.model_propensities,
            self.reward_estimates.cpu().numpy(),
            self.all_action_scores.cpu().numpy(),
            model_values_on_logged_actions,
            maxq_action_idxs,
        )

    def predictor(self) -> DQNPredictor:
        """Builds a DQNPredictor."""
        return DQNPredictor.export(
            self,
            self._actions,
            self.state_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )
