from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gym
import numpy as np

from ml.rl.test.utils import default_normalizer
from ml.rl.training.training_data_page import TrainingDataPage


def create_env(gymenv):
    """
    Creates a gym environment object.

    :param gymenv: String identifier for desired environment.
    """
    if gymenv not in [e.id for e in gym.envs.registry.all()]:
        raise Exception(
            "Warning: Env {} not fount in OpenAI Gym. Quitting.".format(gymenv)
        )
    env = gym.make(gymenv)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    return env, action_dim, state_dim


class OpenAIGymEnvironment:
    def __init__(self, gymenv, epsilon=0.2, max_replay_memory_size=10000):
        """
        Creates an OpenAIGymEnvironment object.

        :param gymenv: String identifier for desired environment.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param max_replay_memory_size: Upper bound on the number of transitions
            to store in replay memory.
        """
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

    def sample_memories(self, batch_size):
        """
        Samples transitions from replay memory uniformly at random.

        :param replay_memroy: Array of transitions.
        :param batch_size: Number of sampled transitions to return.
        """
        cols = [[], [], [], [], [], [], []]
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        return cols

    def get_training_data_page(self, num_samples):
        """
        Returns a TrainingDataPage with shuffled, transformed transitions from
        replay memory.

        :param num_samples: Number of transitions to sample from replay memory.
        """
        states, actions, rewards, next_states, next_actions, terminals,\
            possible_next_actions = self.sample_memories(num_samples)
        return TrainingDataPage(
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_actions, dtype=np.float32),
            np.array(possible_next_actions, dtype=np.float32),
            None, None, np.logical_not(terminals, dtype=np.bool)
        )

    @property
    def normalization(self):
        return default_normalizer(self.state_features)

    def policy(self, trainer, next_state, test):
        """
        Selects the next action.

        :param trainer: RLTrainer object whose policy to follow.
        :param next_state: State to evaluate trainer's policy on.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        """
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
        """
        Inserts transition into replay memory in such a way that retrieving
        transitions uniformly at random will be equivalent to reservoir sampling.
        """
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

    def run_episode(self, trainer, test=False, render=False):
        """
        Runs an episode of the environment. Inserts transitions into replay
        memory and returns the sum of rewards experienced in the episode.

        :param trainer: RLTrainer object whose policy to follow.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        terminal = False
        next_state = self.env.reset()
        next_action = self.policy(trainer, next_state, test)
        reward_sum = 0

        while not terminal:
            state = next_state
            action = next_action
            action_index = np.argmax(action)

            if render:
                self.env.render()

            next_state, reward, terminal, _ = self.env.step(action_index)
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
