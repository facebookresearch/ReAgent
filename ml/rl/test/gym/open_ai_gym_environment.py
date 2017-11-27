from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np
import gym

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


class OpenAIGymEnvironment:
    def __init__(self, gymenv, epsilon=0.2, max_replay_memory_size=10000):
        self.epsilon = epsilon
        self.replay_memory = []
        self.max_replay_memory_size = max_replay_memory_size
        self.memory_num = 0
        self.skip_insert_until = self.max_replay_memory_size

        self.env, self.action_dim, self.state_dim = create_env(gymenv)
        self.state_features = [str(sf) for sf in range(self.state_dim)]
        self.actions = [str(a) for a in range(self.action_dim)]

    @property
    def requires_discrete_actions(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

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
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_actions, dtype=np.float32),
            True - np.array(terminals, dtype=np.bool),
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

    def policy(self, trainer, next_state, test):
        action = np.zeros([self.action_dim], dtype=np.float32)
        if not test and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            action_idx = trainer.get_policy(next_state)
        action[action_idx] = 1.0
        return action

    def insert_into_memory(
        self, state, action, reward, next_state, next_action, terminal,
        possible_next_actions
    ):
        item = (
            state, action, reward, next_state, next_action, terminal,
            possible_next_actions
        )

        if self.memory_num < self.max_replay_memory_size:
            self.replay_memory.append(item)
        elif self.memory_num >= self.skip_insert_until:
            p = float(self.max_replay_memory_size) / self.memory_num
            self.skip_insert_until += np.random.geometric(p)
            rand_index = np.random.randint(self.max_replay_memory_size)
            self.replay_memory[rand_index] = item
        self.memory_num += 1

    def run_episode(self, trainer, test=False):
        terminal = False
        next_state = self.env.reset()
        next_action = self.policy(trainer, next_state, test)
        reward_sum = 0

        while not terminal:
            state = next_state
            action = next_action
            action_index = np.argmax(action)

            next_state_unprocessed, reward, terminal, _ = self.env.step(
                action_index
            )
            next_state = next_state_unprocessed
            next_action = self.policy(trainer, next_state, test)
            reward_sum += reward

            possible_next_actions = [
                0 if terminal else 1 for __ in range(self.action_dim)
            ]

            self.insert_into_memory(
                state, action, reward, next_state, next_action, terminal,
                possible_next_actions
            )

        return reward_sum
