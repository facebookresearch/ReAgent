#!/usr/bin/env python3

import enum

import gym
import numpy as np

from caffe2.python import workspace

from ml.rl.test.utils import default_normalizer
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymDQNPredictorPytorch,
)
from ml.rl.training.training_data_page import TrainingDataPage


class ModelType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    PARAMETRIC_ACTION = "parametric"
    CONTINUOUS_ACTION = "continuous"
    PYTORCH_DISCRETE_DQN = "pytorch_discrete_dqn"


class EnvType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    CONTINUOUS_ACTION = "continuous"


class OpenAIGymEnvironment:
    def __init__(self, gymenv, epsilon, softmax_policy, max_replay_memory_size):
        """
        Creates an OpenAIGymEnvironment object.

        :param gymenv: String identifier for desired environment.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param softmax_policy: 1 to use softmax selection policy or 0 to use
            max q selection.
        :param max_replay_memory_size: Upper bound on the number of transitions
            to store in replay memory.
        """
        self.epsilon = epsilon
        self.softmax_policy = softmax_policy
        self.replay_memory = []
        self.max_replay_memory_size = max_replay_memory_size
        self.memory_num = 0
        self.skip_insert_until = self.max_replay_memory_size

        self._create_env(gymenv)
        if not self.img:
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        if self.action_type == EnvType.DISCRETE_ACTION:
            self.actions = [str(a) for a in range(self.action_dim)]

    def _create_env(self, gymenv):
        """
        Creates a gym environment object and checks if it is supported. We
        support environments that supply Box(x, ) state representations and
        require Discrete(y) or Box(y,) action inputs.

        :param gymenv: String identifier for desired environment.
        """
        if gymenv not in [e.id for e in gym.envs.registry.all()]:
            raise Exception("Env {} not found in OpenAI Gym.".format(gymenv))
        self.env = gym.make(gymenv)

        supports_state = isinstance(self.env.observation_space, gym.spaces.Box) and len(
            self.env.observation_space.shape
        ) in [1, 3]
        supports_action = type(self.env.action_space) in (
            gym.spaces.Discrete,
            gym.spaces.Box,
        )

        if not supports_state and supports_action:
            raise Exception(
                "Unsupported environment state or action type: {}, {}".format(
                    self.env.observation_space, self.env.action_space
                )
            )

        self.action_space = self.env.action_space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = EnvType.DISCRETE_ACTION
            self.action_dim = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = EnvType.CONTINUOUS_ACTION
            self.action_dim = self.env.action_space.shape[0]

        if len(self.env.observation_space.shape) == 1:
            self.state_dim = self.env.observation_space.shape[0]
            self.img = False
        elif len(self.env.observation_space.shape) == 3:
            self.height, self.width, self.num_input_channels = (
                self.env.observation_space.shape
            )
            self.img = True

    def sample_memories(self, batch_size, model_type):
        """
        Samples transitions from replay memory uniformly at random.

        :param batch_size: Number of sampled transitions to return.
        :param model_type: Model type (discrete, parametric).
        """
        cols = [[], [], [], [], [], [], [], [], []]
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)

        if model_type == ModelType.PARAMETRIC_ACTION.value:
            possible_next_actions = []
            for pna_matrix in cols[6]:
                for row in pna_matrix:
                    possible_next_actions.append(row)
        else:
            possible_next_actions = cols[6]

        return TrainingDataPage(
            states=np.array(cols[0], dtype=np.float32),
            actions=np.array(cols[1], dtype=np.float32),
            propensities=None,
            rewards=np.array(cols[2], dtype=np.float32),
            next_states=np.array(cols[3], dtype=np.float32),
            next_actions=np.array(cols[4], dtype=np.float32),
            possible_next_actions=np.array(possible_next_actions, dtype=np.float32),
            reward_timelines=None,
            not_terminals=np.logical_not(np.array(cols[5]), dtype=np.bool),
            time_diffs=np.array(cols[8], dtype=np.int32),
            possible_next_actions_lengths=np.array(cols[7], dtype=np.int32),
        )

    def sample_and_load_training_data_c2(self, num_samples, model_type):
        """
        Loads and preprocesses shuffled, transformed transitions from
        replay memory into the training net.

        :param num_samples: Number of transitions to sample from replay memory.
        :param model_type: Model type (discrete, parametric).
        """
        tdp = self.sample_memories(num_samples, model_type)
        workspace.FeedBlob("states", tdp.states)
        workspace.FeedBlob("actions", tdp.actions)
        workspace.FeedBlob("rewards", tdp.rewards.reshape(-1, 1))
        workspace.FeedBlob("next_states", tdp.next_states)
        workspace.FeedBlob("not_terminals", tdp.not_terminals.reshape(-1, 1))
        workspace.FeedBlob("time_diff", tdp.time_diffs.reshape(-1, 1))
        workspace.FeedBlob("next_actions", tdp.next_actions)
        workspace.FeedBlob("possible_next_actions", tdp.possible_next_actions)
        workspace.FeedBlob(
            "possible_next_actions_lengths", tdp.possible_next_actions_lengths
        )

    @property
    def normalization(self):
        if self.img:
            return None
        else:
            return default_normalizer(self.state_features)

    @property
    def normalization_action(self):
        return default_normalizer(
            [x for x in list(range(self.state_dim, self.state_dim + self.action_dim))]
        )

    def policy(self, predictor, next_state, test):
        """
        Selects the next action.

        :param predictor: RLPredictor object whose policy to follow.
        :param next_state: State to evaluate predictor's policy on.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        """
        # Add a dimension since this expects a batch of examples
        next_state = np.expand_dims(next_state.astype(np.float32), axis=0)
        action = np.zeros([self.action_dim], dtype=np.float32)

        if isinstance(predictor, (GymDQNPredictor, GymDQNPredictorPytorch)):
            if not test and np.random.rand() < self.epsilon:
                action_idx = np.random.randint(self.action_dim)
            else:
                if self.softmax_policy:
                    action_idx = predictor.policy(next_state)[1]
                else:
                    action_idx = predictor.policy(next_state)[0]

            action[action_idx] = 1.0
            return action
        elif isinstance(predictor, GymDDPGPredictor):
            if test:
                return predictor.policy(next_state)[0]
            return predictor.policy(next_state, add_action_noise=True)[0]
        else:
            raise NotImplementedError("Unknown predictor type")

    def insert_into_memory(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        terminal,
        possible_next_actions,
        possible_next_actions_lengths,
        time_diff,
    ):
        """
        Inserts transition into replay memory in such a way that retrieving
        transitions uniformly at random will be equivalent to reservoir sampling.
        """
        item = (
            state,
            action,
            reward,
            next_state,
            next_action,
            terminal,
            possible_next_actions,
            possible_next_actions_lengths,
            time_diff,
        )

        if self.memory_num < self.max_replay_memory_size:
            self.replay_memory.append(item)
        elif self.memory_num >= self.skip_insert_until:
            p = float(self.max_replay_memory_size) / self.memory_num
            self.skip_insert_until += np.random.geometric(p)
            rand_index = np.random.randint(self.max_replay_memory_size)
            self.replay_memory[rand_index] = item
        self.memory_num += 1

    def run_ep_n_times(self, n, predictor, max_steps=None, test=False, render=False):
        """
        Runs an episode of the environment n times and returns the average
        sum of rewards.

        :param n: Number of episodes to average over.
        :param predictor: RLPredictor object whose policy to follow.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        reward_sum = 0.0
        for _ in range(n):
            ep_rew_sum = self.run_episode(predictor, max_steps, test, render)
            reward_sum += ep_rew_sum
        avg_rewards = round(reward_sum / n, 2)
        return avg_rewards

    def transform_state(self, state):
        if self.img:
            # Convert from Height-Width-Channel into Channel-Height-Width
            state = np.transpose(state, axes=[2, 0, 1])
        return state

    def run_episode(self, predictor, max_steps=None, test=False, render=False):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: RLPredictor object whose policy to follow.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        terminal = False
        next_state = self.transform_state(self.env.reset())
        next_action = self.policy(predictor, next_state, test)
        reward_sum = 0
        num_steps_taken = 0

        while not terminal:
            action = next_action
            if render:
                self.env.render()

            if self.action_type == EnvType.DISCRETE_ACTION:
                action_index = np.argmax(action)
                next_state, reward, terminal, _ = self.env.step(action_index)
            else:
                next_state, reward, terminal, _ = self.env.step(action)

            next_state = self.transform_state(next_state)
            num_steps_taken += 1
            next_action = self.policy(predictor, next_state, test)
            reward_sum += reward

            if max_steps and num_steps_taken >= max_steps:
                break

        self.env.reset()
        return reward_sum
