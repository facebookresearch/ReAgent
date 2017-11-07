from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np
import gym
import random

from ml.rl.preprocessing.normalization import NormalizationParameters


def create_env(gymenv):
    if gymenv not in [e.id for e in gym.envs.registry.all()]:
        raise Exception(
            "Warning: Env {} not fount in openai gym, quit.".format(gymenv)
        )
    env = gym.make(gymenv)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    return env, action_dim, state_dim


def sample_memories(replay_memory, batch_size):
    cols = [[], [], [], [], [], [], []]
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    return cols


def max_q(predictor, state, actions, features, action_dim):
    predict_input = dict(
        [(f, np.array([state[i]])) for i, f in enumerate(features)]
    )
    q_values = predictor.predict(predict_input)
    best_value = None
    best_action = None
    for action, key in enumerate(actions):
        q_action = q_values[key][0][0]
        if best_value is None or q_action > best_value:
            best_value = q_action
            best_action = action
    action_mask = np.zeros([action_dim], dtype=np.float32)
    action_mask[best_action] = 1.0
    return action_mask


class OpenAIGymEnvironment:
    def __init__(self, epsilon=0.2, replay_memory_size=10000):
        self.epsilon = epsilon
        self.replay_memory = collections.deque([], maxlen=replay_memory_size)
        self.replay_memory.clear()

        self.env, self.action_dim, self.state_dim = create_env(self.GYMENV)
        self.state_features = [str(sf) for sf in range(self.state_dim)]
        self.actions = [str(a) for a in range(self.action_dim)]

    def get_replay_samples(self, batch_size):
        """
        Returns shuffled, transformed transitions from replay memory.

        Returns:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, state_dim,)
            next_actions: (batch_size,)
            terminals: (batch_size,)
            possible_next_actions: (batch_size, action_dim)
            reward_timelines: None
            evaluator: None
        """
        states, actions, rewards, next_states, next_actions, terminals,\
            possible_next_actions = sample_memories(self.replay_memory, batch_size)
        return (
            np.array(states, dtype=np.float32), np.array(
                actions, dtype=np.float32
            ), np.array(rewards, dtype=np.float32), np.array(
                next_states, dtype=np.float32
            ), np.array(next_actions,
                        dtype=np.float32), np.array(terminals, dtype=np.bool),
            np.array(possible_next_actions, dtype=np.float32), None, None
        )

    @property
    def normalization(self):
        return collections.OrderedDict(
            [
                (
                    state_feature, NormalizationParameters(
                        feature_type="CONTINUOUS",
                        boxcox_lambda=None,
                        boxcox_shift=0,
                        mean=0,
                        stddev=1
                    )
                ) for state_feature in self.state_features
            ]
        )

    @property
    def normalization_action(self):
        return collections.OrderedDict(
            [
                (
                    action, NormalizationParameters(
                        feature_type="CONTINUOUS",
                        boxcox_lambda=None,
                        boxcox_shift=0,
                        mean=0,
                        stddev=1
                    )
                ) for action in self.actions
            ]
        )

    def policy(self, predictor, next_state, test):
        if not test and np.random.rand() < self.epsilon:
            action = np.zeros([self.action_dim], dtype=np.float32)
            action[np.random.randint(self.action_dim)] = 1.0
            return action
        return max_q(
            predictor, next_state, self.actions, self.state_features,
            self.action_dim
        )

    def run_episode(self, predictor, test=False):
        terminal = False
        next_state = self.env.reset()
        next_action = self.policy(predictor, next_state, test)
        reward_sum = 0

        while not terminal:
            state = next_state
            action = next_action
            action_index = np.argmax(action)

            next_state_unprocessed, reward, terminal, _ = self.env.step(
                action_index
            )
            next_state = next_state_unprocessed
            next_action = self.policy(predictor, next_state, test)
            reward_sum += reward

            possible_next_actions = [
                0 if terminal else 1 for __ in range(self.action_dim)
            ]

            self.replay_memory.append(
                (
                    state, action, reward, next_state, next_action, terminal,
                    possible_next_actions
                )
            )

        return reward_sum


class CartpoleV0Environment(OpenAIGymEnvironment):
    GYMENV = 'CartPole-v0'


class CartpoleV1Environment(OpenAIGymEnvironment):
    GYMENV = 'CartPole-v1'
