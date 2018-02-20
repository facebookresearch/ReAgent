#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from caffe2.python import core, workspace

from ml.rl.preprocessing.caffe_utils import dict_list_to_blobs
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.test.utils import default_normalizer

# Environment parameters
DISCOUNT = 0.9
EPSILON = 0.1

# Used for legible specification of grids
W = 1  # Walls
S = 2  # Starting position
G = 3  # Goal position


class GridworldBase(object):
    """Implements a simple grid world, a domain often used as a very simple
    to solve benchmark for reinforcement learning algorithms, also see:
    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/lecture-notes/lecture11-6pp.pdf
    """

    # Must be overridden by derived class
    ACTIONS: List[str] = ['L', 'R', 'U', 'D']

    grid = np.array(
        [
            [S, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, G],  #
        ]
    )

    width = 5
    height = 5
    STATES = list(range(width * height))

    USING_ONLY_VALID_ACTION = True

    transition_noise = 0.01  # nonzero means stochastic transition

    def __init__(self):
        self.reset()
        self._optimal_policy = self._compute_optimal()

    @property
    def normalization(self):
        return default_normalizer(self.STATES)

    def _compute_optimal(self):
        not_visited = {
            (y, x)
            for x in range(self.width) for y in range(self.height)
        }
        queue = collections.deque()
        queue.append(tuple(j[0] for j in np.where(self.grid == G)))
        policy = np.empty(self.grid.shape, dtype=np.object)
        print("INITIAL POLICY")
        print(policy)
        while len(queue) > 0:
            current = queue.pop()
            if current in not_visited:
                not_visited.remove(current)

            possible_actions = self.possible_next_actions(
                self._index(current), True
            )
            for action in possible_actions:
                self._state = self._index(current)
                next_state, _, _, _ = self.step(action)
                next_state_pos = self._pos(next_state)
                if next_state_pos not in not_visited:
                    continue
                not_visited.remove(next_state_pos)
                if not self.is_terminal(next_state) and \
                        self.grid[next_state_pos] != W:
                    policy[next_state_pos] = self.invert_action(action)
                    queue.appendleft(self._pos(next_state))
        print("FINAL POLICY")
        print(policy)
        return policy

    def invert_action(self, action: str) -> str:
        if action == 'U':
            return 'D'
        if action == 'R':
            return 'L'
        if action == 'D':
            return 'U'
        if action == 'L':
            return 'R'
        raise Exception("Invalid action", action)

    def _index(self, pos):
        y, x = pos
        return x * self.height + y
        # 0 5 10 15 20
        # 1 6 11 16 ...
        # y is row index, x is col index

    def _pos(self, state):
        return state % self.height, state // self.height

    def reset(self):
        self._state = self._index(np.argwhere(self.grid == S)[0])
        return self._state

    def reward(self, state):
        return 1 if self.grid[self._pos(state)] == G else 0

    def transition_probabilities(self, state, action):
        y, x = self._pos(state)
        probabilities = np.zeros((self.width * self.height, ))
        left_state = self._index((y, max(0, x - 1)))
        right_state = self._index((y, min(self.width - 1, x + 1)))
        down_state = self._index((min(self.height - 1, y + 1), x))
        up_state = self._index((max(0, y - 1), x))
        left_state, right_state, down_state, up_state = \
            [state if self.grid[self._pos(s)] == W else s
             for s in [left_state, right_state, down_state, up_state]]
        if action == 'L':
            probabilities[left_state] += 1 - 2 * self.transition_noise
            probabilities[up_state] += self.transition_noise
            probabilities[down_state] += self.transition_noise
        elif action == 'R':
            probabilities[right_state] += 1 - 2 * self.transition_noise
            probabilities[up_state] += self.transition_noise
            probabilities[down_state] += self.transition_noise
        elif action == 'U':
            probabilities[up_state] += 1 - 2 * self.transition_noise
            probabilities[left_state] += self.transition_noise
            probabilities[right_state] += self.transition_noise
        elif action == 'D':
            probabilities[down_state] += 1 - 2 * self.transition_noise
            probabilities[left_state] += self.transition_noise
            probabilities[right_state] += self.transition_noise
        else:
            raise Exception("Invalid action", action)
        return probabilities

    def _no_cheat_step(self, state, action: str) -> int:
        p = self.transition_probabilities(state, action)
        return np.random.choice(self.size, p=p)

    def step(self, action: str,
             with_possible=True) -> Tuple[int, float, bool, List[str]]:
        self._state = self._no_cheat_step(self._state, action)
        reward = self.reward(self._state)
        if with_possible:
            possible_next_action = self.possible_next_actions(self._state)
        else:
            possible_next_action = None
        return self._state, reward, self.is_terminal(self._state),\
            possible_next_action

    def optimal_policy(self, state):
        y, x = self._pos(state)
        return self._optimal_policy[y, x]

    def sample_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            possible_actions = self.possible_next_actions(state)
            if len(possible_actions) == 0:
                return None
            return np.random.choice(possible_actions)
        else:
            return self.optimal_policy(state)

    @property
    def num_actions(self):
        return len(self.ACTIONS)

    @property
    def num_states(self):
        return self.width * self.height

    @property
    def size(self):
        return self.width * self.height

    def is_terminal(self, state):
        return self.grid[self._pos(state)] == G

    def is_valid(self, y, x):
        return x >= 0 and x <= self.width - 1 and y >= 0 and y <= self.height - 1

    def move_on_pos(self, y, x, act):
        if act == 'L':
            return y, x - 1
        elif act == 'R':
            return y, x + 1
        elif act == 'U':
            return y - 1, x
        elif act == 'D':
            return y + 1, x
        else:
            raise Exception("Invalid action", act)

    def move_on_pos_limit(self, y, x, act):
        y, x = self.move_on_pos(y, x, act)
        x = min(max(0, x), self.width - 1)
        y = min(max(0, y), self.height - 1)
        return y, x

    def move_on_index_limit(self, state, act):
        y, x = self._pos(state)
        y, x = self.move_on_pos_limit(y, x, act)
        return self._index((y, x))

    def possible_next_actions(self, state, ignore_terminal=False):
        possible_actions = []
        if not ignore_terminal and self.is_terminal(state):
            return possible_actions
        # else: #Q learning is better with true possible_next_actions
        #    return range(len(self.ACTIONS))
        y, x = self._pos(state)
        if x - 1 >= 0:
            possible_actions.append('L')
        if x + 1 <= self.width - 1:
            possible_actions.append('R')
        if y - 1 >= 0:
            possible_actions.append('U')
        if y + 1 <= self.height - 1:
            possible_actions.append('D')
        return possible_actions

    def q_transition_matrix(self, assume_optimal_policy):
        T = np.zeros((self.size, self.size))
        for state in range(self.size):
            if not self.is_terminal(state):
                poss_a = self.ACTIONS
                if self.USING_ONLY_VALID_ACTION:
                    poss_a = self.possible_next_actions(state)
                poss_a_count = len(poss_a)
                fraction = 1.0 / poss_a_count
                for action in poss_a:
                    transition_probabilities = self.transition_probabilities(
                        state, action
                    )
                    if assume_optimal_policy:
                        if action != self.optimal_policy(state):
                            continue
                        action_probability = 1.0
                    else:
                        action_probability = fraction
                    T[state,
                      :] += \
                        action_probability * \
                        transition_probabilities
        return T

    def reward_vector(self):
        return np.array([self.reward(s) for s in range(self.size)])

    def true_q_values(self, discount, assume_optimal_policy):
        R = self.reward_vector()
        print("REWARD VECTOR")
        print(R)
        T = self.q_transition_matrix(assume_optimal_policy)
        print('T:', T)
        return np.linalg.solve(np.eye(self.size, self.size) - (discount * T), R)

    def true_values_for_sample(
        self, states, actions, assume_optimal_policy: bool
    ):
        true_q_values = self.true_q_values(DISCOUNT, assume_optimal_policy)
        print("TRUE Q")
        print(true_q_values.reshape([5, 5]))
        results = []
        for x in range(len(states)):
            int_state = int(list(states[x].keys())[0])
            next_state = self.move_on_index_limit(int_state, actions[x])
            if self.is_terminal(int_state):
                results.append(self.reward(int_state))
            else:
                results.append(
                    self.reward(int_state) +
                    (DISCOUNT * true_q_values[next_state])
                )
        return results

    def generate_samples_discrete(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[List[Dict[int, float]], List[str], List[float], List[
        Dict[int, float]
    ], List[str], List[bool], List[List[str]], List[Dict[int, float]]]:
        states = []
        actions: List[str] = []
        rewards = []
        next_states = []
        next_actions: List[str] = []
        is_terminals = []
        state: int = -1
        is_terminal = True
        next_action = None
        possible_next_actions: List[List[str]] = []
        transition = 0
        last_terminal = -1
        reward_timelines = []
        while True:
            if is_terminal:
                if transition >= num_transitions:
                    break
                state = self.reset()
                is_terminal = False
                action = self.sample_policy(state, epsilon)
            else:
                action = next_action

            next_state, reward, is_terminal, possible_next_action = self.step(
                action, with_possible
            )
            next_action = self.sample_policy(next_state, epsilon)
            if next_action is None:
                next_action = ''

            states.append({state: 1.0})
            actions.append(action)
            rewards.append(reward)
            next_states.append({next_state: 1.0})
            next_actions.append(next_action)
            is_terminals.append(is_terminal)
            possible_next_actions.append(possible_next_action)
            if not is_terminal:
                reward_timelines.append({0: 0.0})
            else:
                reward_timelines.append({0: 1.0})
                i = 1
                while transition - i > last_terminal:
                    assert len(reward_timelines[transition - i]) == 1
                    reward_timelines[transition - i][i] = 1.0
                    i += 1
                last_terminal = transition

            state = next_state
            transition += 1

        return (
            states, actions, rewards, next_states, next_actions, is_terminals,
            possible_next_actions, reward_timelines
        )

    def preprocess_samples_discrete(
        self,
        states: List[Dict[int, float]],
        actions: List[str],
        rewards: List[float],
        next_states: List[Dict[int, float]],
        next_actions: List[str],
        is_terminals: List[bool],
        possible_next_actions: List[List[str]],
        reward_timelines: Optional[List[Dict[int, float]]],
        minibatch_size: int,
    ) -> List[TrainingDataPage]:
        # Shuffle
        if reward_timelines is None:
            merged = list(
                zip(
                    states, actions, rewards, next_states, next_actions,
                    is_terminals, possible_next_actions
                )
            )
            random.shuffle(merged)
            states, actions, rewards, next_states, next_actions, \
                is_terminals, possible_next_actions = zip(*merged)
        else:
            merged = list(
                zip(
                    states, actions, rewards, next_states, next_actions,
                    is_terminals, possible_next_actions, reward_timelines
                )
            )
            random.shuffle(merged)
            states, actions, rewards, next_states, next_actions, \
                is_terminals, possible_next_actions, reward_timelines = zip(*merged)

        net = core.Net('gridworld_preprocessing')
        preprocessor = PreprocessorNet(net, True)
        lengths, keys, values = dict_list_to_blobs(states, 'states')
        state_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization,
            'state_norm',
        )
        lengths, keys, values = dict_list_to_blobs(next_states, 'next_states')
        next_state_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization,
            'next_state_norm',
        )
        workspace.RunNetOnce(net)
        actions_one_hot = np.zeros(
            [len(actions), len(self.ACTIONS)], dtype=np.float32
        )
        for i, action in enumerate(actions):
            actions_one_hot[i, self.ACTIONS.index(action)] = 1
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_actions_one_hot = np.zeros(
            [len(next_actions), len(self.ACTIONS)], dtype=np.float32
        )
        for i, action in enumerate(next_actions):
            if action == '':
                continue
            next_actions_one_hot[i, self.ACTIONS.index(action)] = 1
        possible_next_actions_mask = []
        for pna in possible_next_actions:
            pna_mask = [0] * self.num_actions
            for action in pna:
                pna_mask[self.ACTIONS.index(action)] = 1
            possible_next_actions_mask.append(pna_mask)
        possible_next_actions_mask = np.array(
            possible_next_actions_mask, dtype=np.float32
        )
        is_terminals = np.array(is_terminals, dtype=np.bool).reshape(-1, 1)
        if reward_timelines is not None:
            reward_timelines = np.array(reward_timelines, dtype=np.object)

        states_ndarray = workspace.FetchBlob(state_matrix)
        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        tdps = []
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            tdps.append(
                TrainingDataPage(
                    states=states_ndarray[start:end],
                    actions=actions_one_hot[start:end],
                    rewards=rewards[start:end],
                    next_states=next_states_ndarray[start:end],
                    next_actions=next_actions_one_hot[start:end],
                    possible_next_actions=possible_next_actions_mask[start:end],
                    reward_timelines=reward_timelines[start:end]
                    if reward_timelines is not None else None,
                )
            )
        return tdps

    def generate_samples(
        self,
        num_transitions,
        epsilon,
        with_possible=True,
    ):
        raise NotImplementedError()

    def preprocess_samples(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        is_terminals,
        possible_next_actions,
        reward_timelines,
        minibatch_size,
    ):
        raise NotImplementedError()
