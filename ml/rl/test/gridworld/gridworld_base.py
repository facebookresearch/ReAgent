#!/usr/bin/env python3


import collections
import multiprocessing
import random
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.test.utils import default_normalizer
from ml.rl.training.training_data_page import TrainingDataPage


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
        "propensities",
        "rewards",
        "possible_actions",
        "next_states",
        "next_actions",
        "terminals",
        "possible_next_actions",
        "reward_timelines",
    ]

    def __init__(
        self,
        mdp_ids: List[int],
        sequence_numbers: List[int],
        states: List[Dict[int, float]],
        actions: List[str],
        propensities: List[float],
        rewards: List[float],
        possible_actions: List[List[str]],
        next_states: List[Dict[int, float]],
        next_actions: List[str],
        terminals: List[bool],
        possible_next_actions: List[List[str]],
        reward_timelines: List[Dict[int, float]],
    ) -> None:
        self.mdp_ids = mdp_ids
        self.sequence_numbers = sequence_numbers
        self.states = states
        self.actions = actions
        self.propensities = propensities
        self.rewards = rewards
        self.possible_actions = possible_actions
        self.next_states = next_states
        self.next_actions = next_actions
        self.terminals = terminals
        self.possible_next_actions = possible_next_actions
        self.reward_timelines = reward_timelines

    def shuffle(self):
        # Shuffle
        if self.reward_timelines is None:
            merged = list(
                zip(
                    self.mdp_ids,
                    self.sequence_numbers,
                    self.states,
                    self.actions,
                    self.propensities,
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
                self.propensities,
                self.rewards,
                self.possible_actions,
                self.next_states,
                self.next_actions,
                self.terminals,
                self.possible_next_actions,
            ) = zip(*merged)
        else:
            merged = list(
                zip(
                    self.mdp_ids,
                    self.sequence_numbers,
                    self.states,
                    self.actions,
                    self.propensities,
                    self.rewards,
                    self.possible_actions,
                    self.next_states,
                    self.next_actions,
                    self.terminals,
                    self.possible_next_actions,
                    self.reward_timelines,
                )
            )
            random.shuffle(merged)
            (
                self.mdp_ids,
                self.sequence_numbers,
                self.states,
                self.actions,
                self.propensities,
                self.rewards,
                self.possible_actions,
                self.next_states,
                self.next_actions,
                self.terminals,
                self.possible_next_actions,
                self.reward_timelines,
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

    @property
    def normalization(self):
        return default_normalizer(self.STATES)

    def action_to_index(self, action):
        return self.ACTIONS.index(action)

    def index_to_action(self, index):
        return self.ACTIONS[index]

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
        return np.random.choice(self.size, p=p)

    def step(
        self, action: str, with_possible=True
    ) -> Tuple[int, float, bool, List[str]]:
        self._state = self._no_cheat_step(self._state, action)
        reward = self.reward(self._state)
        if with_possible:
            possible_next_action = self.possible_actions(self._state)
        else:
            possible_next_action = None
        return self._state, reward, self.is_terminal(self._state), possible_next_action

    def optimal_policy(self, state):
        y, x = self._pos(state)
        return self._optimal_policy[y, x]

    def sample_policy(self, state, epsilon) -> Tuple[str, float]:
        possible_actions = self.possible_actions(state)
        if len(possible_actions) == 0:
            return "", 1.0
        if np.random.rand() < epsilon:
            if len(possible_actions) == 1:
                return possible_actions[0], 1.0
            else:
                return (
                    np.random.choice(possible_actions),
                    epsilon / (len(possible_actions) - 1),
                )
        else:
            return self.optimal_policy(state), (1.0 - epsilon)

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

    def generate_samples_discrete(  # multiprocessing
        self, num_transitions, epsilon, with_possible=True
    ) -> Samples:
        sub_transitions = int(num_transitions / 8)
        args = [sub_transitions] * 8
        func = partial(
            self.generate_samples_discrete_internal,
            epsilon=epsilon,
            with_possible=with_possible,
        )
        pool = multiprocessing.Pool(processes=8)
        res = pool.map(func, args)

        merge = Samples(
            mdp_ids=[],
            sequence_numbers=[],
            states=[],
            actions=[],
            propensities=[],
            rewards=[],
            possible_actions=[],
            next_states=[],
            next_actions=[],
            terminals=[],
            possible_next_actions=[],
            reward_timelines=[],
        )
        for s in res:
            merge.mdp_ids += s.mdp_ids
            merge.sequence_numbers += s.sequence_numbers
            merge.states += s.states
            merge.actions += s.actions
            merge.propensities += s.propensities
            merge.rewards += s.rewards
            merge.possible_actions += s.possible_actions
            merge.next_states += s.next_states
            merge.next_actions += s.next_actions
            merge.terminals += s.terminals
            merge.possible_next_actions += s.possible_next_actions
            merge.reward_timelines += s.reward_timelines
        return merge

    def generate_samples_discrete_internal(
        self, num_transitions, epsilon, with_possible=True
    ) -> Samples:
        states = []
        actions: List[str] = []
        propensities = []
        rewards = []
        next_states = []
        next_actions: List[str] = []
        terminals = []
        mdp_ids = []
        sequence_numbers = []
        state: int = -1
        terminal = True
        next_action = ""
        next_propensity = 1.0
        possible_actions: List[List[str]] = []
        possible_next_actions: List[List[str]] = []
        transition = 0
        last_terminal = -1
        reward_timelines = []
        mdp_id = -1
        sequence_number = 0
        while True:
            if terminal:
                if transition >= num_transitions:
                    break
                state = self.reset()
                terminal = False
                mdp_id += 1
                sequence_number = 0
                action, propensity = self.sample_policy(state, epsilon)
            else:
                action = next_action
                propensity = next_propensity
                sequence_number += 1

            if with_possible:
                possible_action = self.possible_actions(state)

            next_state, reward, terminal, possible_next_action = self.step(
                action, with_possible
            )
            next_action, next_propensity = self.sample_policy(next_state, epsilon)

            mdp_ids.append(mdp_id)
            sequence_numbers.append(sequence_number)
            states.append({state: 1.0})
            actions.append(action)
            propensities.append(propensity)
            rewards.append(reward)
            possible_actions.append(possible_action)
            next_states.append({next_state: 1.0})
            next_actions.append(next_action)
            terminals.append(terminal)
            possible_next_actions.append(possible_next_action)
            if not terminal:
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
        return Samples(
            mdp_ids=mdp_ids,
            sequence_numbers=sequence_numbers,
            states=states,
            actions=actions,
            propensities=propensities,
            rewards=rewards,
            possible_actions=possible_actions,
            next_states=next_states,
            next_actions=next_actions,
            terminals=terminals,
            possible_next_actions=possible_next_actions,
            reward_timelines=reward_timelines,
        )

    def preprocess_samples_discrete(
        self, samples: Samples, minibatch_size: int, one_hot_action: bool = True
    ) -> List[TrainingDataPage]:
        samples.shuffle()

        net = core.Net("gridworld_preprocessing")
        C2.set_net(net)
        preprocessor = PreprocessorNet(True)
        saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
        state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            "state_norm",
            False,
            False,
            False,
        )
        saa = StackedAssociativeArray.from_dict_list(samples.next_states, "next_states")
        next_state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            "next_state_norm",
            False,
            False,
            False,
        )
        workspace.RunNetOnce(net)
        actions_one_hot = np.zeros(
            [len(samples.actions), len(self.ACTIONS)], dtype=np.float32
        )
        for i, action in enumerate(samples.actions):
            actions_one_hot[i, self.action_to_index(action)] = 1
        actions = np.array(
            [self.action_to_index(action) for action in samples.actions], dtype=np.int64
        )
        rewards = np.array(samples.rewards, dtype=np.float32).reshape(-1, 1)
        propensities = np.array(samples.propensities, dtype=np.float32).reshape(-1, 1)
        next_actions_one_hot = np.zeros(
            [len(samples.next_actions), len(self.ACTIONS)], dtype=np.float32
        )
        for i, action in enumerate(samples.next_actions):
            if action == "":
                continue
            next_actions_one_hot[i, self.action_to_index(action)] = 1
        possible_next_actions_mask = []
        for pna in samples.possible_next_actions:
            pna_mask = [0] * self.num_actions
            for action in pna:
                pna_mask[self.action_to_index(action)] = 1
            possible_next_actions_mask.append(pna_mask)
        possible_next_actions_mask = np.array(
            possible_next_actions_mask, dtype=np.float32
        )
        terminals = np.array(samples.terminals, dtype=np.bool).reshape(-1, 1)
        not_terminals = np.logical_not(terminals)
        episode_values = None
        if samples.reward_timelines is not None:
            episode_values = np.zeros(rewards.shape, dtype=np.float32)
            for i, reward_timeline in enumerate(samples.reward_timelines):
                for time_diff, reward in reward_timeline.items():
                    episode_values[i, 0] += reward * (DISCOUNT ** time_diff)

        preprocessor = Preprocessor(self.normalization, False)

        states_ndarray = workspace.FetchBlob(state_matrix)
        states_ndarray = preprocessor.forward(states_ndarray).numpy()

        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        next_states_ndarray = preprocessor.forward(next_states_ndarray).numpy()

        time_diffs = np.ones(len(states_ndarray))
        tdps = []
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            tdps.append(
                TrainingDataPage(
                    states=states_ndarray[start:end],
                    actions=actions_one_hot[start:end]
                    if one_hot_action
                    else actions[start:end],
                    propensities=propensities[start:end],
                    rewards=rewards[start:end],
                    next_states=next_states_ndarray[start:end],
                    not_terminals=not_terminals[start:end],
                    next_actions=next_actions_one_hot[start:end],
                    possible_next_actions=possible_next_actions_mask[start:end],
                    episode_values=episode_values[start:end]
                    if episode_values is not None
                    else None,
                    time_diffs=time_diffs[start:end],
                )
            )
        return tdps

    def generate_samples(self, num_transitions, epsilon, with_possible=True):
        raise NotImplementedError()

    def preprocess_samples(self, samples, minibatch_size):
        raise NotImplementedError()
