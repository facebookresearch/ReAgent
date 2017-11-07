#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np
import six

from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.rl_trainer import RLTrainer


class LimitedActionDiscreteActionTrainer(DiscreteActionTrainer):
    def __init__(
        self,
        normalization_parameters,
        parameters,
    ):
        self._quantile_states = collections.deque(
            maxlen=parameters.action_budget.window_size
        )
        self._quantile = 100 - parameters.action_budget.action_limit
        self.quantile_value = 0
        self._limited_action = np.argmax(
            np.array(parameters.actions) ==
            parameters.action_budget.limited_action
        )
        self._discount_factor = parameters.rl.gamma
        self._quantile_update_rate = \
            parameters.action_budget.quantile_update_rate
        self._quantile_update_frequency = \
            parameters.action_budget.quantile_update_frequency
        self._update_counter = 0
        super(self.__class__,
              self).__init__(normalization_parameters, parameters)
        self._max_q = parameters.rl.maxq_learning

    def get_maxq_actions(self, states, possible_next_actions):
        """
        Takes in an array of states and outputs an array of the same shape whose
        ith entry is the action a that maximizes Q(state_i, a). Only considers
        Q values of possible actions.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        """
        q_values = self.get_q_values(states, possible_next_actions, False)
        # TODO: Speed this up
        q_values_mask = np.zeros(
            [states.shape[0], self.num_actions], dtype=np.float32
        )
        for i, a in enumerate(q_values):
            q_values_mask[i] = a
        return q_values_mask

    def action_values(self, states, action_idx):
        return self.get_q_values(states, None, False)[:, action_idx]

    def stream(
        self, states, actions, rewards, next_states, next_actions, is_terminals,
        possible_next_actions, reward_timelines, evaluator
    ):
        self._quantile_states.extendleft(states)
        if self._update_counter % self._quantile_update_frequency == \
                self._quantile_update_frequency - 1:
            self._update_quantile()
        self._update_counter += 1

        if self._max_q:
            q_next_actions = self.get_maxq_actions(
                next_states, possible_next_actions
            )
        else:
            q_next_actions = next_actions
        penalty = self._reward_penalty(actions, q_next_actions, is_terminals)
        RLTrainer.stream(
            self, states, actions, rewards - penalty, next_states, next_actions,
            is_terminals, possible_next_actions, None, evaluator
        )

    def _update_quantile(self):
        states = np.array(self._quantile_states, dtype=np.float32)
        limited_action_values = self.action_values(states, self._limited_action)
        base_action_values = np.max(
            np.array(
                [
                    self.action_values(states, action)
                    for action in six.moves.range(self.num_actions)
                    if action != self._limited_action
                ]
            ),
            axis=0
        )
        target = np.percentile(
            limited_action_values - base_action_values, self._quantile
        )
        print("REWARD PENALTY TARGET:", target)
        self.quantile_value += self._quantile_update_rate * target
        print("QUANTILE:", self.quantile_value)

    def _reward_penalty(self, actions, next_actions, is_terminals):
        return (
            (
                (actions[:, self._limited_action] > 0.999) -
                self._discount_factor * (1 - is_terminals) *
                (next_actions[:, self._limited_action] > 0.999)
            ) * self.quantile_value
        ).astype(np.float32)
