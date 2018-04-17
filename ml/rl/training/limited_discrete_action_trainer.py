#!/usr/bin/env python3


from typing import Dict, Any
import collections
import numpy as np
import six

from caffe2.python import workspace

from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.rl_trainer import RLTrainer


class LimitedActionDiscreteActionTrainer(DiscreteActionTrainer):
    def __init__(
        self,
        parameters,
        normalization_parameters: Dict[int, NormalizationParameters],
    ) -> None:
        self._quantile_states: Any = collections.deque(
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
        DiscreteActionTrainer.__init__(
            self,
            parameters,
            normalization_parameters,
        )
        self._max_q = parameters.rl.maxq_learning

    def action_values(self, states, action_idx):
        workspace.FeedBlob('states', states)
        workspace.RunNetOnce(self.all_q_score_model.net)
        q = workspace.FetchBlob(self.all_q_score_output)
        return q[:, action_idx]

    def train_numpy(self, tdp, evaluator):
        self._quantile_states.extendleft(tdp.states)
        if self._update_counter % self._quantile_update_frequency == \
                self._quantile_update_frequency - 1:
            self._update_quantile()
        self._update_counter += 1

        if self._max_q:
            workspace.FeedBlob('states', tdp.states)
            workspace.FeedBlob('actions', tdp.possible_next_actions)
            workspace.RunNetOnce(self.q_score_model.net)
            q_values = workspace.FetchBlob(self.q_score_output)
            q_next_actions = np.argmax(q_values, axis=1).reshape(-1, 1)
            q_next_actions_mask = np.zeros(
                [q_next_actions.shape[0], self.num_actions], dtype=np.float32
            )
            for x in range(q_next_actions.shape[0]):
                q_next_actions_mask[q_next_actions[x, 0], 0] = 1.0
            q_next_actions = q_next_actions_mask
        else:
            q_next_actions = tdp.next_actions
        penalty = self._reward_penalty(
            tdp.actions, q_next_actions, tdp.not_terminals
        )
        assert penalty.shape == tdp.rewards.shape, "" + str(
            penalty.shape
        ) + "" + str(tdp.rewards.shape)
        tdp.rewards = tdp.rewards - penalty
        RLTrainer.train_numpy(
            self,
            tdp,
            evaluator,
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

    def _reward_penalty(self, actions, next_actions, not_terminals):
        print(not_terminals)
        print(not_terminals.shape)
        print(actions.shape)
        print(next_actions.shape)
        return (
            (
                (actions[:, self._limited_action] > 0.999) -
                self._discount_factor * (not_terminals[:, 0]) *
                (next_actions[:, self._limited_action] > 0.999)
            ) * self.quantile_value
        ).astype(np.float32).reshape(-1, 1)
