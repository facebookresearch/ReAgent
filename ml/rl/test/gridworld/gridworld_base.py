#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import collections
import itertools
import logging
import random
from typing import List, Tuple

import numpy as np
import torch
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import Caffe2SparseToDenseProcessor
from ml.rl.test.base.utils import default_normalizer
from ml.rl.test.environment.environment import (
    ACTION,
    Environment,
    Samples,
    shuffle_samples,
)
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


# Environment parameters
DISCOUNT = 0.9

# Used for legible specification of grids
W = 1  # Walls
S = 2  # Starting position
G = 3  # Goal position


class GridworldBase(Environment):
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

    width = grid.shape[1]
    height = grid.shape[0]
    STATES = list(range(width * height))

    USING_ONLY_VALID_ACTION = True

    transition_noise = 0.01  # nonzero means stochastic transition

    REWARD_SCALE = 10.0

    def __init__(self):
        self.reset()
        self._optimal_policy = self._compute_optimal()
        self._optimal_actions = self._compute_optimal_actions()
        self.sparse_to_dense_net = None
        self._true_q_values = collections.defaultdict(dict)
        self._true_q_epsilon_values = collections.defaultdict(dict)

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
            possible_actions = self.possible_actions(
                self._index(current), ignore_terminal=True
            )
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
        print("OPTIMAL ACTIONS:")
        print(optimal_actions)
        print()
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

            possible_actions = self.possible_actions(
                self._index(current), ignore_terminal=True
            )
            for action in possible_actions:
                next_state_pos = self.move_on_pos(current[0], current[1], action)
                next_state = self._index(next_state_pos)
                if next_state_pos not in not_visited:
                    continue
                not_visited.remove(next_state_pos)
                if not self.is_terminal(next_state) and self.grid[next_state_pos] != W:
                    policy[next_state_pos] = self.invert_action(action)
                    queue.appendleft(self._pos(next_state))
        print("OPTIMAL POLICY (NON UNIQUE):")
        print(policy)
        print()
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
        return self.REWARD_SCALE if self.grid[self._pos(state)] == G else 0

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

    def step(self, action) -> Tuple[int, float, bool, object]:
        self._state = self._no_cheat_step(self._state, action)
        reward = self.reward(self._state)
        # make results similar to OpenAI gym
        info = None
        return self._state, reward, self.is_terminal(self._state), info

    def _process_state(self, state):
        processed_state = {state: 1.0}
        return processed_state

    def optimal_policy(self, state):
        y, x = self._pos(state)
        return self._optimal_policy[y, x]

    def policy_probabilities(self, state, epsilon):
        """Returns the probabilities of the epsilon-greedy version of the optimal
        policy for a given state.
        """
        probabilities = np.zeros(len(self.ACTIONS))
        state = int(list(state.keys())[0])
        possible_actions = self.possible_actions(state)
        if len(possible_actions) == 0:
            return probabilities
        optimal_action = self.optimal_policy(state)
        for action in possible_actions:
            if action == optimal_action:
                action_probability = (1.0 - epsilon) + epsilon / len(possible_actions)
            else:
                action_probability = epsilon / len(possible_actions)
            probabilities[self.action_to_index(action)] = action_probability
        return probabilities

    def policy_probabilities_for_sample(self, states, epsilon):
        """Returns the probabilities of the epsilon-greedy version of the optimal
        policy for a vector of states.
        """
        results = []
        for x in range(len(states)):
            results.append(self.policy_probabilities(states[x], epsilon))
        return np.array(results)

    def sample_policy(
        self, state, use_continuous_action, epsilon
    ) -> Tuple[ACTION, ACTION, float]:
        """
        Sample an action following epsilon-greedy
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        possible_actions = self.possible_actions(state)
        if len(possible_actions) == 0:
            if use_continuous_action:
                return "", {}, 1.0
            return "", "", 1.0

        optimal_action = self.optimal_policy(state)
        if random.random() < epsilon:
            action: ACTION = random.choice(possible_actions)
        else:
            action = optimal_action
        if action == optimal_action:
            action_probability = (1.0 - epsilon) + epsilon / len(possible_actions)
        else:
            action_probability = epsilon / len(possible_actions)
        if use_continuous_action:
            processed_action: ACTION = self.action_to_features(action)
        else:
            processed_action = action
        return action, processed_action, action_probability

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

    def is_wall(self, state):
        return self.grid[self._pos(state)] == W

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

    def possible_actions(
        self,
        state,
        terminal=False,
        ignore_terminal=False,
        use_continuous_action: bool = False,
        **kwargs,
    ) -> List[ACTION]:
        possible_actions: List[ACTION] = []
        if not ignore_terminal and self.is_terminal(state):
            return possible_actions
        assert not self.is_wall(state)
        # else: #Q learning is better with true possible_next_actions
        #    return range(len(self.ACTIONS))
        y, x = self._pos(state)
        if x - 1 >= 0 and not self.is_wall(self._index((y, x - 1))):
            possible_actions.append("L")
        if x + 1 <= self.width - 1 and not self.is_wall(self._index((y, x + 1))):
            possible_actions.append("R")
        if y - 1 >= 0 and not self.is_wall(self._index((y - 1, x))):
            possible_actions.append("U")
        if y + 1 <= self.height - 1 and not self.is_wall(self._index((y + 1, x))):
            possible_actions.append("D")
        if use_continuous_action:
            return [self.action_to_features(pa) for pa in possible_actions]
        return possible_actions

    def q_transition_matrix(self, assume_optimal_policy, epsilon=0.0):
        T = np.zeros((self.size, self.size))
        for state in range(self.size):
            if not self.is_terminal(state) and not self.is_wall(state):
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
                            action_probability = epsilon / len(poss_a)
                        else:
                            action_probability = (1.0 - epsilon) + epsilon / len(poss_a)
                    else:
                        action_probability = fraction
                    T[state, :] += action_probability * transition_probabilities

        return T

    def reward_vector(self):
        return np.array([self.reward(s) for s in range(self.size)])

    def true_q_values(self, discount, assume_optimal_policy):
        if (
            self._true_q_values.get(discount) is not None
            and self._true_q_values.get(discount).get(assume_optimal_policy) is not None
        ):
            return self._true_q_values[discount][assume_optimal_policy]

        R = self.reward_vector()
        T = self.q_transition_matrix(assume_optimal_policy)
        self._true_q_values[discount][assume_optimal_policy] = np.linalg.solve(
            np.eye(self.size, self.size) - (discount * T), R
        )
        print("TRUE STATE VALUES ASSUMING OPTIMAL = {}:".format(assume_optimal_policy))
        print(
            self._true_q_values[discount][assume_optimal_policy].reshape(
                self.height, self.width
            )
        )
        return self._true_q_values[discount][assume_optimal_policy]

    def true_q_epsilon_values(self, discount, epsilon):
        if (
            self._true_q_epsilon_values.get(discount) is not None
            and self._true_q_epsilon_values.get(discount).get(epsilon) is not None
        ):
            return self._true_q_epsilon_values[discount][epsilon]

        R = self.reward_vector()
        T = self.q_transition_matrix(True, epsilon)
        self._true_q_epsilon_values[discount][epsilon] = np.linalg.solve(
            np.eye(self.size, self.size) - (discount * T), R
        )
        print("TRUE STATE VALUES FOR OPTIMAL WITH EPSILON = {}:".format(epsilon))
        print(
            self._true_q_epsilon_values[discount][epsilon].reshape(
                self.height, self.width
            )
        )
        return self._true_q_epsilon_values[discount][epsilon]

    def true_values_for_sample(self, states, actions, assume_optimal_policy: bool):
        true_q_values = self.true_q_values(DISCOUNT, assume_optimal_policy)
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

    def true_epsilon_values_for_sample(self, states, actions, epsilon):
        true_q_epsilon_values = self.true_q_epsilon_values(DISCOUNT, epsilon)
        results = []
        for x in range(len(states)):
            int_state = int(list(states[x].keys())[0])
            next_state = self.move_on_index_limit(int_state, actions[x])
            if self.is_terminal(next_state):
                results.append(self.reward(next_state))
            else:
                results.append(
                    self.reward(next_state)
                    + (DISCOUNT * true_q_epsilon_values[next_state])
                )
        return np.array(results).reshape(-1, 1)

    def true_epsilon_values_all_actions_for_sample(self, states, epsilon):
        """Returns the true values of the epsilon-greedy optimal policy for
         *all actions* for each state in 'states.'
        """
        results = np.zeros((len(states), len(self.ACTIONS)))
        for x in range(len(self.ACTIONS)):
            action = self.ACTIONS[x]
            actions_vec = [action for _ in range(len(states))]
            results[:, x] = self.true_epsilon_values_for_sample(
                states, actions_vec, epsilon
            )[:, 0]
        return results

    def true_rewards_for_sample(self, states, actions):
        """Returns the true rewards for each state/action pair in states/actions.
        """
        results = []
        for x in range(len(states)):
            int_state = int(list(states[x].keys())[0])
            next_state = self.move_on_index_limit(int_state, actions[x])
            results.append(self.reward(next_state))
        return np.array(results).reshape(-1, 1)

    def true_rewards_all_actions_for_sample(self, states):
        """Returns the true rewards for for *all actions* of
        each state in 'states.'
        """
        results = np.zeros((len(states), len(self.ACTIONS)))
        for x in range(len(self.ACTIONS)):
            action = self.ACTIONS[x]
            actions_vec = [action for _ in range(len(states))]
            results[:, x] = self.true_rewards_for_sample(states, actions_vec)[:, 0]
        return results

    def preprocess_samples_discrete(
        self,
        samples: Samples,
        minibatch_size: int,
        one_hot_action: bool = True,
        use_gpu: bool = False,
        do_shuffle: bool = True,
    ) -> List[TrainingDataPage]:

        if do_shuffle:
            logger.info("Shuffling...")
            samples = shuffle_samples(samples)

        logger.info("Preprocessing...")
        sparse_to_dense_processor = Caffe2SparseToDenseProcessor()

        if self.sparse_to_dense_net is None:
            self.sparse_to_dense_net = core.Net("gridworld_sparse_to_dense")
            C2.set_net(self.sparse_to_dense_net)
            saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
            sorted_features, _ = sort_features_by_normalization(self.normalization)
            self.state_matrix, _ = sparse_to_dense_processor(sorted_features, saa)
            saa = StackedAssociativeArray.from_dict_list(
                samples.next_states, "next_states"
            )
            self.next_state_matrix, _ = sparse_to_dense_processor(sorted_features, saa)
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
        logger.info("Converting PA to Torch...")
        possible_action_strings = np.array(
            list(itertools.zip_longest(*samples.possible_actions, fillvalue=""))
        ).T
        possible_actions_mask = torch.zeros([len(samples.actions), len(self.ACTIONS)])
        for i, action in enumerate(self.ACTIONS):
            possible_actions_mask[:, i] = torch.tensor(
                np.max(possible_action_strings == action, axis=1).astype(np.int64)
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
                possible_actions_mask=possible_actions_mask[start:end],
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
