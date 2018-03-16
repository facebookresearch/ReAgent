from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gym
import numpy as np

from ml.rl.test.utils import default_normalizer
from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor
from ml.rl.training.training_data_page import TrainingDataPage


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

        self._create_env(gymenv)
        if not self.img:
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        self.actions = [str(a) for a in range(self.action_dim)]

    def _create_env(self, gymenv):
        """
        Creates a gym environment object and checks if it is supported. We
        support environments that supply Box(x, ) state representations and
        require Discrete(y) action inputs.

        :param gymenv: String identifier for desired environment.
        """
        if gymenv not in [e.id for e in gym.envs.registry.all()]:
            raise Exception("Env {} not found in OpenAI Gym.".format(gymenv))
        self.env = gym.make(gymenv)

        supports_state = isinstance(
            self.env.observation_space, gym.spaces.Box
        ) and len(self.env.observation_space.shape) in [1, 3]
        supports_action = isinstance(self.env.action_space, gym.spaces.Discrete)

        if not supports_state and supports_action:
            raise Exception(
                "Unsupported environment state or action type: {}, {}".format(
                    self.env.observation_space, self.env.action_space
                )
            )

        self.action_dim = self.env.action_space.n

        if (len(self.env.observation_space.shape) == 1):
            self.state_dim = self.env.observation_space.shape[0]
            self.img = False
        elif len(self.env.observation_space.shape) == 3:
            self.height, self.width, self.num_input_channels = \
                self.env.observation_space.shape
            self.img = True

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
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(next_actions, dtype=np.float32),
            np.array(possible_next_actions, dtype=np.float32), None,
            np.logical_not(terminals, dtype=np.bool).reshape(-1, 1)
        )

    @property
    def normalization(self):
        return default_normalizer(self.state_features)

    def policy(self, predictor, next_state, test):
        """
        Selects the next action.

        :param predictor: RLPredictor object whose policy to follow.
        :param next_state: State to evaluate predictor's policy on.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        """
        next_state = next_state.astype(np.float32).reshape(1, -1)
        action = np.zeros([self.action_dim], dtype=np.float32)
        if not test and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            # Convert next_state to a list[dict[int,float]]
            next_state_dict = [{}]
            for i in range(next_state.shape[1]):
                next_state_dict[0][i] = next_state[0][i]
            # For DiscreteActionPredictors use the output policy directly
            if isinstance(predictor, DiscreteActionPredictor):
                action_str = predictor.discrete_action_policy(next_state_dict)[1]
                action_idx = self.actions.index(action_str.decode("utf-8"))
            else:
                action_score_dict = predictor.predict(next_state_dict)[0]
                best_action = None
                best_score = None
                for action_name, action_score in action_score_dict.items():
                    if best_action is None or best_score < action_score:
                        best_action = action_name
                        best_score = action_score
                action_idx = self.actions.index(best_action)
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

    def run_episode(self, predictor, test=False, render=False):
        """
        Runs an episode of the environment. Inserts transitions into replay
        memory and returns the sum of rewards experienced in the episode.

        :param predictor: RLPredictor object whose policy to follow.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        terminal = False
        next_state = self.env.reset()
        next_action = self.policy(predictor, next_state, test)
        reward_sum = 0

        while not terminal:
            state = next_state
            action = next_action
            action_index = np.argmax(action)

            if render:
                self.env.render()

            next_state, reward, terminal, _ = self.env.step(action_index)
            next_action = self.policy(predictor, next_state, test)
            reward_sum += reward

            possible_next_actions = [
                0 if terminal else 1 for __ in range(self.action_dim)
            ]

            self.insert_into_memory(
                state, action, reward, next_state, next_action, terminal,
                possible_next_actions
            )

        return reward_sum
