#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import collections
import itertools
import logging
import random
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import sparse_to_dense
from ml.rl.test.utils import default_normalizer
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


# Environment parameters
DISCOUNT = 0.9

# Used for legible specification of grids
W = 1  # Walls
S = 2  # Starting position
G = 3  # Goal position


class Samples(object):
    __slots__ = [
        "mdp_ids",
        "sequence_numbers",
        "states",
        "actions",
        "action_probabilities",
        "rewards",
        "possible_actions",
        "next_states",
        "next_actions",
        "terminals",
        "possible_next_actions",
    ]

    def __init__(
        self,
        mdp_ids: List[str],
        sequence_numbers: List[int],
        states: List[Dict[int, float]],
        actions,  # List[str] if discrete or Dict[str,float] if parametric
        action_probabilities: List[float],
        rewards: List[float],
        possible_actions,  # List[str] if discrete or List[Dict[str,float]] if parametric
        next_states: List[Dict[int, float]],
        next_actions: List[str],
        terminals: List[bool],
        possible_next_actions: List[List[str]],
    ) -> None:
        self.mdp_ids = mdp_ids
        self.sequence_numbers = sequence_numbers
        self.states = states
        self.actions = actions
        self.action_probabilities = action_probabilities
        self.rewards = rewards
        self.possible_actions = possible_actions
        self.next_states = next_states
        self.next_actions = next_actions
        self.terminals = terminals
        self.possible_next_actions = possible_next_actions

    def shuffle(self):
        # Shuffle
        merged = list(
            zip(
                self.mdp_ids,
                self.sequence_numbers,
                self.states,
                self.actions,
                self.action_probabilities,
                self.rewards,
                self.possible_actions,
                self.next_states,
                self.next_actions,
                self.terminals,
                self.possible_next_actions,
            )
        )
        random.shuffle(merged)
        (
            self.mdp_ids,
            self.sequence_numbers,
            self.states,
            self.actions,
            self.action_probabilities,
            self.rewards,
            self.possible_actions,
            self.next_states,
            self.next_actions,
            self.terminals,
            self.possible_next_actions,
        ) = zip(*merged)


class GridworldBase(object):
    """Implements a simple grid world, a domain often used as a very simple
    to solve benchmark for reinforcement learning algorithms, also see:
    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/lecture-notes/lecture11-6pp.pdf
    """

    # Must be overridden by derived class
    ACTIONS: List[str] = ["L", "R", "U", "D"]

    grid = np.array(
        [
            [S, 0, 0, 0, 0],  #
            [0, 0, W, 0, 0],  #
            [0, 0, W, 0, 0],  #
            [0, 0, W, 0, 0],  #
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
        self._optimal_actions = self._compute_optimal_actions()
        self.sparse_to_dense_net = None

    @property
    def normalization(self):
        return default_normalizer(self.STATES)

    def action_to_index(self, action):
        return self.ACTIONS.index(action)

    def index_to_action(self, index):
        return self.ACTIONS[index]

    def _compute_optimal_actions(self):
        not_visited = {(y, x) for x in range(self.width) for y in range(self.height)}
        queue = collections.deque()
        bfs_start = tuple(j[0] for j in np.where(self.grid == G))
        queue.append(bfs_start)
        working_set = set()
        working_set.add(bfs_start)
        # record optimal actions for each grid cell
        optimal_actions = np.empty(self.grid.shape, dtype=np.object)
        while len(queue) > 0:
            current = queue.pop()
            working_set.remove(current)
            if current in not_visited:
                not_visited.remove(current)
            possible_actions = self.possible_actions(self._index(current), True)
            for action in possible_actions:
                next_state_pos = self.move_on_pos(current[0], current[1], action)
                next_state = self._index(next_state_pos)
                if next_state_pos not in not_visited:
                    continue
                if not self.is_terminal(next_state) and self.grid[next_state_pos] != W:
                    self._add_optimal_action(
                        optimal_actions, next_state_pos, self.invert_action(action)
                    )
                    bfs_next = self._pos(next_state)
                    if bfs_next not in working_set:
                        queue.appendleft(bfs_next)
                        working_set.add(bfs_next)
        print("OPTIMAL ACTIONS")
        print(optimal_actions)
        return optimal_actions

    def _add_optimal_action(self, optimal_actions, pos, action):
        if optimal_actions[pos] is None:
            optimal_actions[pos] = [action]
        else:
            optimal_actions[pos].append(action)

    def _compute_optimal(self):
        not_visited = {(y, x) for x in range(self.width) for y in range(self.height)}
        queue = collections.deque()
        queue.append(tuple(j[0] for j in np.where(self.grid == G)))
        policy = np.empty(self.grid.shape, dtype=np.object)
        while len(queue) > 0:
            current = queue.pop()
            if current in not_visited:
                not_visited.remove(current)

            possible_actions = self.possible_actions(self._index(current), True)
            for action in possible_actions:
                next_state_pos = self.move_on_pos(current[0], current[1], action)
                next_state = self._index(next_state_pos)
                if next_state_pos not in not_visited:
                    continue
                not_visited.remove(next_state_pos)
                if not self.is_terminal(next_state) and self.grid[next_state_pos] != W:
                    policy[next_state_pos] = self.invert_action(action)
                    queue.appendleft(self._pos(next_state))
        print("OPTIMAL POLICY")
        print(policy)
        return policy

    def invert_action(self, action: str) -> str:
        if action == "U":
            return "D"
        if action == "R":
            return "L"
        if action == "D":
            return "U"
        if action == "L":
            return "R"
        raise Exception("Invalid action", action)

    def _index(self, pos):
        y, x = pos
        return x + y * self.width
        # 0 1 2 3 4
        # 5 6 7 8 9
        # ...
        # y is row index, x is col index

    def _pos(self, state):
        return state // self.width, state % self.width

    def reset(self):
        self._state = self._index(np.argwhere(self.grid == S)[0])
        return self._state

    def reward(self, state):
        return 1 if self.grid[self._pos(state)] == G else 0

    def transition_probabilities(self, state, action):
        y, x = self._pos(state)
        probabilities = np.zeros((self.width * self.height,))
        left_state = self._index((y, max(0, x - 1)))
        right_state = self._index((y, min(self.width - 1, x + 1)))
        down_state = self._index((min(self.height - 1, y + 1), x))
        up_state = self._index((max(0, y - 1), x))
        left_state, right_state, down_state, up_state = [
            state if self.grid[self._pos(s)] == W else s
            for s in [left_state, right_state, down_state, up_state]
        ]
        if action == "L":
            probabilities[left_state] += 1 - 2 * self.transition_noise
            probabilities[up_state] += self.transition_noise
            probabilities[down_state] += self.transition_noise
        elif action == "R":
            probabilities[right_state] += 1 - 2 * self.transition_noise
            probabilities[up_state] += self.transition_noise
            probabilities[down_state] += self.transition_noise
        elif action == "U":
            probabilities[up_state] += 1 - 2 * self.transition_noise
            probabilities[left_state] += self.transition_noise
            probabilities[right_state] += self.transition_noise
        elif action == "D":
            probabilities[down_state] += 1 - 2 * self.transition_noise
            probabilities[left_state] += self.transition_noise
            probabilities[right_state] += self.transition_noise
        else:
            raise Exception("Invalid action", action)
        return probabilities

    def _no_cheat_step(self, state, action: str) -> int:
        p = self.transition_probabilities(state, action)
        return int(np.random.choice(self.size, p=p))

    def step(self, action: str) -> Tuple[int, float, bool, List[str]]:
        self._state = self._no_cheat_step(self._state, action)
        reward = self.reward(self._state)
        possible_next_action = self.possible_actions(self._state)
        return self._state, reward, self.is_terminal(self._state), possible_next_action

    def optimal_policy(self, state):
        y, x = self._pos(state)
        return self._optimal_policy[y, x]

    def sample_policy(self, state, epsilon) -> Tuple[str, float]:
        possible_actions = self.possible_actions(state)
        if len(possible_actions) == 0:
            return "", 1.0
        optimal_action = self.optimal_policy(state)
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
        else:
            action = optimal_action
        if action == optimal_action:
            action_probability = (1.0 - epsilon) + epsilon / len(possible_actions)
        else:
            action_probability = epsilon / len(possible_actions)
        return action, action_probability

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
        if act == "L":
            return y, x - 1
        elif act == "R":
            return y, x + 1
        elif act == "U":
            return y - 1, x
        elif act == "D":
            return y + 1, x
        else:
            raise Exception("Invalid action", act)

    def move_on_pos_limit(self, y, x, act):
        original_y, original_x = y, x
        y, x = self.move_on_pos(y, x, act)
        x = min(max(0, x), self.width - 1)
        y = min(max(0, y), self.height - 1)
        if self.grid[y, x] == W:
            return original_y, original_x  # Bumped into wall
        return y, x

    def move_on_index_limit(self, state, act):
        y, x = self._pos(state)
        y, x = self.move_on_pos_limit(y, x, act)
        return self._index((y, x))

    def possible_actions(self, state, ignore_terminal=False):
        possible_actions = []
        if not ignore_terminal and self.is_terminal(state):
            return possible_actions
        # else: #Q learning is better with true possible_next_actions
        #    return range(len(self.ACTIONS))
        y, x = self._pos(state)
        if x - 1 >= 0:
            possible_actions.append("L")
        if x + 1 <= self.width - 1:
            possible_actions.append("R")
        if y - 1 >= 0:
            possible_actions.append("U")
        if y + 1 <= self.height - 1:
            possible_actions.append("D")
        return possible_actions

    def q_transition_matrix(self, assume_optimal_policy):
        T = np.zeros((self.size, self.size))
        for state in range(self.size):
            if not self.is_terminal(state):
                poss_a = self.ACTIONS
                if self.USING_ONLY_VALID_ACTION:
                    poss_a = self.possible_actions(state)
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
                    T[state, :] += action_probability * transition_probabilities
        return T

    def reward_vector(self):
        return np.array([self.reward(s) for s in range(self.size)])

    def true_q_values(self, discount, assume_optimal_policy):
        R = self.reward_vector()
        T = self.q_transition_matrix(assume_optimal_policy)
        return np.linalg.solve(np.eye(self.size, self.size) - (discount * T), R)

    def true_values_for_sample(self, states, actions, assume_optimal_policy: bool):
        true_q_values = self.true_q_values(DISCOUNT, assume_optimal_policy)
        print("TRUE VALUES ASSUMING OPTIMAL: ", assume_optimal_policy)
        print(true_q_values.reshape(5, 5))
        results = []
        for x in range(len(states)):
            int_state = int(list(states[x].keys())[0])
            next_state = self.move_on_index_limit(int_state, actions[x])
            if self.is_terminal(next_state):
                results.append(self.reward(next_state))
            else:
                results.append(
                    self.reward(next_state) + (DISCOUNT * true_q_values[next_state])
                )
        return np.array(results).reshape(-1, 1)

    def true_rewards_for_sample(self, states, actions):
        results = []
        for x in range(len(states)):
            int_state = int(list(states[x].keys())[0])
            next_state = self.move_on_index_limit(int_state, actions[x])
            results.append(self.reward(next_state))
        return np.array(results).reshape(-1, 1)

    def generate_samples_discrete(
        self, num_transitions, epsilon, discount_factor
    ) -> Samples:
        # Initialize lists
        states: List[Dict[int, float]] = [{}] * num_transitions
        actions: List[str] = [""] * num_transitions
        action_probabilities = [0.0] * num_transitions
        rewards = [0.0] * num_transitions
        next_states: List[Dict[int, float]] = [{}] * num_transitions
        next_actions: List[str] = [""] * num_transitions
        terminals = [False] * num_transitions
        mdp_ids = [""] * num_transitions
        sequence_numbers = [0] * num_transitions
        possible_actions: List[List[str]] = [[]] * num_transitions
        possible_next_actions: List[List[str]] = [[]] * num_transitions

        state: int = -1
        terminal = True
        next_action = ""
        next_action_probability = 1.0
        transition = 0
        mdp_id = -1
        sequence_number = 0

        # We run until we finish the episode that completes N transitions, but
        # we may have to go beyond N to reach the end of that episode
        while not terminal or transition < num_transitions:
            if terminal:
                state = self.reset()
                terminal = False
                mdp_id += 1
                sequence_number = 0
                action, action_probability = self.sample_policy(state, epsilon)
            else:
                action = next_action
                action_probability = next_action_probability
                sequence_number += 1

            possible_action = self.possible_actions(state)

            next_state, reward, terminal, possible_next_action = self.step(action)
            next_action, next_action_probability = self.sample_policy(
                next_state, epsilon
            )

            # We want exactly N data points, but we need to wait until the
            # episode is over so we can get the episode values.  This function
            # will set episode values if they are in the range [0,N) and ignore
            # otherwise.
            def set_if_in_range(l, i, v):
                if i >= num_transitions:
                    return
                l[i] = v

            def add_if_in_range(l, i, v):
                if i >= num_transitions:
                    return
                l[i] += v

            set_if_in_range(mdp_ids, transition, str(mdp_id))
            set_if_in_range(sequence_numbers, transition, sequence_number)
            set_if_in_range(states, transition, {int(state): 1.0})
            set_if_in_range(actions, transition, action)
            set_if_in_range(action_probabilities, transition, action_probability)
            set_if_in_range(rewards, transition, reward)
            set_if_in_range(possible_actions, transition, possible_action)
            set_if_in_range(next_states, transition, {int(next_state): 1.0})
            set_if_in_range(next_actions, transition, next_action)
            set_if_in_range(terminals, transition, terminal)
            set_if_in_range(possible_next_actions, transition, possible_next_action)

            state = next_state
            transition += 1
        return Samples(
            mdp_ids=mdp_ids,
            sequence_numbers=sequence_numbers,
            states=states,
            actions=actions,
            action_probabilities=action_probabilities,
            rewards=rewards,
            possible_actions=possible_actions,
            next_states=next_states,
            next_actions=next_actions,
            terminals=terminals,
            possible_next_actions=possible_next_actions,
        )

    def preprocess_samples_discrete(
        self,
        samples: Samples,
        minibatch_size: int,
        one_hot_action: bool = True,
        use_gpu: bool = False,
    ) -> List[TrainingDataPage]:
        logger.info("Shuffling...")
        samples.shuffle()
        logger.info("Preprocessing...")

        if self.sparse_to_dense_net is None:
            self.sparse_to_dense_net = core.Net("gridworld_sparse_to_dense")
            C2.set_net(self.sparse_to_dense_net)
            saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
            sorted_features, _ = sort_features_by_normalization(self.normalization)
            self.state_matrix, _ = sparse_to_dense(
                saa.lengths, saa.keys, saa.values, sorted_features
            )
            saa = StackedAssociativeArray.from_dict_list(
                samples.next_states, "next_states"
            )
            self.next_state_matrix, _ = sparse_to_dense(
                saa.lengths, saa.keys, saa.values, sorted_features
            )
            C2.set_net(None)
        else:
            StackedAssociativeArray.from_dict_list(samples.states, "states")
            StackedAssociativeArray.from_dict_list(samples.next_states, "next_states")
        workspace.RunNetOnce(self.sparse_to_dense_net)

        logger.info("Converting to Torch...")
        actions_one_hot = torch.tensor(
            (np.array(samples.actions).reshape(-1, 1) == np.array(self.ACTIONS)).astype(
                np.int64
            )
        )
        actions = actions_one_hot.argmax(dim=1, keepdim=True)
        rewards = torch.tensor(samples.rewards, dtype=torch.float32).reshape(-1, 1)
        action_probabilities = torch.tensor(
            samples.action_probabilities, dtype=torch.float32
        ).reshape(-1, 1)
        next_actions_one_hot = torch.tensor(
            (
                np.array(samples.next_actions).reshape(-1, 1) == np.array(self.ACTIONS)
            ).astype(np.int64)
        )
        logger.info("Converting PNA to Torch...")
        possible_next_action_strings = np.array(
            list(itertools.zip_longest(*samples.possible_next_actions, fillvalue=""))
        ).T
        possible_next_actions_mask = torch.zeros(
            [len(samples.next_actions), len(self.ACTIONS)]
        )
        for i, action in enumerate(self.ACTIONS):
            possible_next_actions_mask[:, i] = torch.tensor(
                np.max(possible_next_action_strings == action, axis=1).astype(np.int64)
            )
        terminals = torch.tensor(samples.terminals, dtype=torch.int32).reshape(-1, 1)
        not_terminal = 1 - terminals
        logger.info("Converting RT to Torch...")

        time_diffs = torch.ones([len(samples.states), 1])

        logger.info("Preprocessing...")
        preprocessor = Preprocessor(self.normalization, False)

        states_ndarray = workspace.FetchBlob(self.state_matrix)
        states_ndarray = preprocessor.forward(states_ndarray)

        next_states_ndarray = workspace.FetchBlob(self.next_state_matrix)
        next_states_ndarray = preprocessor.forward(next_states_ndarray)

        logger.info("Batching...")
        tdps = []
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            tdp = TrainingDataPage(
                states=states_ndarray[start:end],
                actions=actions_one_hot[start:end]
                if one_hot_action
                else actions[start:end],
                propensities=action_probabilities[start:end],
                rewards=rewards[start:end],
                next_states=next_states_ndarray[start:end],
                not_terminal=not_terminal[start:end],
                next_actions=next_actions_one_hot[start:end],
                possible_next_actions_mask=possible_next_actions_mask[start:end],
                time_diffs=time_diffs[start:end],
            )
            tdp.set_type(torch.cuda.FloatTensor if use_gpu else torch.FloatTensor)
            tdps.append(tdp)
        return tdps

    def generate_samples(self, num_transitions, epsilon, discount_factor):
        raise NotImplementedError()

    def preprocess_samples(self, samples, minibatch_size):
        raise NotImplementedError()
