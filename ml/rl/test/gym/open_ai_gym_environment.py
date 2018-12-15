#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import enum

import gym
import numpy as np
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymSACPredictor,
)
from ml.rl.test.utils import default_normalizer, only_continuous_normalizer
from ml.rl.training.dqn_predictor import DQNPredictor


class ModelType(enum.Enum):
    CONTINUOUS_ACTION = "continuous"
    SOFT_ACTOR_CRITIC = "soft_actor_critic"
    PYTORCH_DISCRETE_DQN = "pytorch_discrete_dqn"
    PYTORCH_PARAMETRIC_DQN = "pytorch_parametric_dqn"


class EnvType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    CONTINUOUS_ACTION = "continuous"


class OpenAIGymEnvironment:
    def __init__(self, gymenv, epsilon=0, softmax_policy=False, gamma=0.99):
        """
        Creates an OpenAIGymEnvironment object.

        :param gymenv: String identifier for desired environment.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param softmax_policy: 1 to use softmax selection policy or 0 to use
            max q selection.
        """
        self.epsilon = epsilon
        self.softmax_policy = softmax_policy
        self.gamma = gamma

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

    @property
    def normalization(self):
        if self.img:
            return None
        else:
            return default_normalizer(self.state_features)

    @property
    def normalization_action(self):
        return only_continuous_normalizer(
            [x for x in list(range(self.state_dim, self.state_dim + self.action_dim))]
        )

    def policy(self, predictor, next_state, test, state_preprocessor=None):
        """
        Selects the next action.

        :param predictor: RLPredictor object whose policy to follow.
        :param next_state: State to evaluate predictor's policy on.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        # Add a dimension since this expects a batch of examples
        next_state = np.expand_dims(next_state.astype(np.float32), axis=0)
        action = np.zeros([self.action_dim], dtype=np.float32)
        action_probability = 1.0

        if isinstance(predictor, GymDQNPredictor):
            if not test and np.random.rand() < self.epsilon:
                action_idx = np.random.randint(self.action_dim)
                action_probability = self.epsilon
            else:
                action_probability = 1.0 - self.epsilon
                if state_preprocessor:
                    next_state = state_preprocessor.forward(next_state)
                if self.softmax_policy:
                    action_idx = predictor.policy(next_state)[1]
                else:
                    action_idx = predictor.policy(next_state)[0]

            action[action_idx] = 1.0
            return action, action_probability
        elif isinstance(predictor, GymDDPGPredictor):
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            if test:
                return predictor.policy(next_state)[0], action_probability
            return (
                predictor.policy(next_state, add_action_noise=True)[0],
                action_probability,
            )
        elif isinstance(predictor, GymSACPredictor):
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            return predictor.policy(next_state)[0], action_probability
        elif isinstance(predictor, DQNPredictor):
            # Use DQNPredictor directly - useful to test caffe2 predictor
            # assumes state preprocessor already part of predictor net.
            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            q_values = predictor.predict(sparse_next_states)
            action_idx = max(q_values[0], key=q_values[0].get)
            action[int(action_idx)] = 1.0
            return action, action_probability
        else:
            raise NotImplementedError("Unknown predictor type")

    def run_ep_n_times(
        self,
        n,
        predictor,
        max_steps=None,
        test=False,
        render=False,
        state_preprocessor=None,
    ):
        """
        Runs an episode of the environment n times and returns the average
        sum of rewards.

        :param n: Number of episodes to average over.
        :param predictor: RLPredictor object whose policy to follow.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        reward_sum = 0.0
        discounted_reward_sum = 0.0
        for _ in range(n):
            ep_rew_sum, ep_raw_discounted_sum = self.run_episode(
                predictor, max_steps, test, render, state_preprocessor
            )
            reward_sum += ep_rew_sum
            discounted_reward_sum += ep_raw_discounted_sum
        avg_rewards = round(reward_sum / n, 2)
        avg_discounted_rewards = round(discounted_reward_sum / n, 2)
        return avg_rewards, avg_discounted_rewards

    def transform_state(self, state):
        if self.img:
            # Convert from Height-Width-Channel into Channel-Height-Width
            state = np.transpose(state, axes=[2, 0, 1])
        return state

    def run_episode(
        self,
        predictor,
        max_steps=None,
        test=False,
        render=False,
        state_preprocessor=None,
    ):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: RLPredictor object whose policy to follow.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        terminal = False
        next_state = self.transform_state(self.env.reset())
        next_action, _ = self.policy(predictor, next_state, test, state_preprocessor)
        reward_sum = 0
        discounted_reward_sum = 0
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
            next_action, _ = self.policy(
                predictor, next_state, test, state_preprocessor
            )
            reward_sum += reward
            discounted_reward_sum += reward * self.gamma ** (num_steps_taken - 1)

            if max_steps and num_steps_taken >= max_steps:
                break

        self.env.reset()
        return reward_sum, discounted_reward_sum
