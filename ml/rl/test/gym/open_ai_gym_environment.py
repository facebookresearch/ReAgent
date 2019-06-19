#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import enum
from functools import reduce
from typing import Dict, Tuple, Union

import gym
import ml.rl.test.gym.pomdp  # noqa
import numpy as np
from gym import Env
from ml.rl.test.base.utils import only_continuous_normalizer
from ml.rl.test.environment.environment import Environment
from ml.rl.test.gym.gym_predictor import GymDQNPredictor, GymPredictor, GymSACPredictor
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.training.rl_predictor_pytorch import RLPredictor


class ModelType(enum.Enum):
    CONTINUOUS_ACTION = "continuous"
    SOFT_ACTOR_CRITIC = "soft_actor_critic"
    PYTORCH_DISCRETE_DQN = "pytorch_discrete_dqn"
    PYTORCH_PARAMETRIC_DQN = "pytorch_parametric_dqn"


class EnvType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    CONTINUOUS_ACTION = "continuous"


class OpenAIGymEnvironment(Environment):
    def __init__(
        self,
        gymenv: Union[str, Env],
        epsilon=0,
        softmax_policy=False,
        gamma=0.99,
        epsilon_decay=1,
        minimum_epsilon=None,
    ):
        """
        Creates an OpenAIGymEnvironment object.

        :param gymenv: String identifier for desired environment or environment
            object itself.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param softmax_policy: 1 to use softmax selection policy or 0 to use
            max q selection.
        :param gamma: Discount rate
        :param epsilon_decay: How much to decay epsilon over each iteration in training.
        :param minimum_epsilon: Lower bound of epsilon.
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.softmax_policy = softmax_policy
        self.gamma = gamma
        self._create_env(gymenv)

        if not self.img:
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        if self.action_type == EnvType.DISCRETE_ACTION:
            self.actions = [str(a + self.state_dim) for a in range(self.action_dim)]

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)

    def _create_env(self, gymenv: Union[str, Env]):
        """
        Creates a gym environment object and checks if it is supported. We
        support environments that supply Box(x, ) state representations and
        require Discrete(y) or Box(y,) action inputs.

        :param gymenv: String identifier for desired environment or environment
            object itself.
        """
        if isinstance(gymenv, Env):
            self.env = gymenv
            self.env_name = gymenv.unwrapped.spec.id
        else:
            if gymenv not in [e.id for e in gym.envs.registry.all()]:
                raise Exception("Env {} not found in OpenAI Gym.".format(gymenv))
            self.env = gym.make(gymenv)
            self.env_name = gymenv

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
            return only_continuous_normalizer(
                list(range(self.state_dim)),
                self.env.observation_space.low,
                self.env.observation_space.high,
            )

    @property
    def normalization_action(self):
        if self.action_type == EnvType.DISCRETE_ACTION:
            return only_continuous_normalizer(
                list(range(self.state_dim, self.state_dim + self.action_dim)),
                min_value=0,
                max_value=1,
            )
        return only_continuous_normalizer(
            list(range(self.state_dim, self.state_dim + self.action_dim)),
            min_value=self.env.action_space.low,
            max_value=self.env.action_space.high,
        )

    def reset(self):
        init_state = self.env.reset()
        assert len(init_state) == self.state_dim
        return init_state

    def step(self, action):
        res = self.env.step(action)
        next_state = res[0]
        assert len(next_state) == self.state_dim
        return res

    def policy(
        self,
        predictor: Union[RLPredictor, GymPredictor, None],
        next_state,
        test,
        state_preprocessor=None,
    ) -> Tuple[np.ndarray, float]:
        """
        Selects the next action.

        :param predictor: RLPredictor/GymPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param next_state: State to evaluate predictor's policy on.
        :param test: Whether or not to bypass exploration (if predictor is not None).
            For discrete action problems, the exploration policy is epsilon-greedy.
            For continuous action problems, the exploration is achieved by adding
            noise to action outputs.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        if predictor is None or (
            not test
            and self.action_type == EnvType.DISCRETE_ACTION
            and np.random.rand() < self.epsilon
        ):
            raw_action, _, action_probability = self.sample_policy(
                state=None, use_continuous_action=False
            )
            if self.action_type == EnvType.DISCRETE_ACTION:
                action: np.ndarray = np.zeros([self.action_dim], dtype=np.float32)
                action[raw_action] = 1
                return action, action_probability
            return raw_action, action_probability

        # Add a dimension since this expects a batch of examples
        next_state = np.expand_dims(next_state.astype(np.float32), axis=0)
        action = np.zeros([self.action_dim], dtype=np.float32)

        if isinstance(predictor, GymDQNPredictor):
            action_probability = 1.0 if test else 1.0 - self.epsilon
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            if self.softmax_policy:
                action_idx = predictor.policy(next_state)[1]
            else:
                action_idx = predictor.policy(next_state)[0]
            action[action_idx] = 1.0
            return action, action_probability
        elif isinstance(predictor, GymSACPredictor):
            # FIXME: need to calculate action probability properly
            # FIXME: also need to support adding noise on outputs when test is True
            action_probability = 0.0
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            return predictor.policy(next_state)[0], action_probability
        elif isinstance(predictor, DQNPredictor):
            action_probability = 1.0 if test else 1.0 - self.epsilon
            # Use DQNPredictor directly - useful to test caffe2 predictor
            # assumes state preprocessor already part of predictor net.
            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            q_values = predictor.predict(sparse_next_states)
            action_idx = int(max(q_values[0], key=q_values[0].get)) - self.state_dim
            action[action_idx] = 1.0
            return action, action_probability
        elif isinstance(predictor, ParametricDQNPredictor):
            # Needs to get a list of candidate actions if actions are continuous
            if self.action_type == EnvType.CONTINUOUS_ACTION:
                raise NotImplementedError()
            action_probability = 1.0 if test else 1.0 - self.epsilon
            next_state = np.repeat(next_state, repeats=self.action_dim, axis=0)
            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            sparse_actions = [
                {str(i + self.state_dim): 1} for i in range(self.action_dim)
            ]
            q_values = predictor.predict(sparse_next_states, sparse_actions)
            q_values = np.fromiter(
                map(lambda x: x["Q"], q_values), np.float  # type: ignore
            ).reshape(self.action_dim)
            action_idx = np.argmax(q_values)
            action[action_idx] = 1.0
            return action, action_probability
        else:
            raise NotImplementedError("Unknown predictor type")

    def run_ep_n_times(
        self,
        n,
        predictor: Union[RLPredictor, GymPredictor, None],
        max_steps=None,
        test=False,
        render=False,
        state_preprocessor=None,
    ):
        """
        Runs an episode of the environment n times and returns the average
        sum of rewards.

        :param n: Number of episodes to average over.
        :param predictor: RLPredictor/GymPredictor object whose policy to
            follow. If set to None, use a random policy
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

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        if self.img:
            # Convert from Height-Width-Channel into Channel-Height-Width
            state = np.transpose(state, axes=[2, 0, 1])
        return state

    def run_episode(
        self,
        predictor: Union[RLPredictor, GymPredictor, None],
        max_steps=None,
        test=False,
        render=False,
        state_preprocessor=None,
    ):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: RLPredictor/GymPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        :param state_preprocessor: State preprocessor to use to preprocess states
        """
        terminal = False
        next_state = self.transform_state(self.reset())
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
                next_state, reward, terminal, _ = self.step(action_index)
            else:
                next_state, reward, terminal, _ = self.step(action)

            next_state = self.transform_state(next_state)
            num_steps_taken += 1
            next_action, _ = self.policy(
                predictor, next_state, test, state_preprocessor
            )
            reward_sum += reward
            discounted_reward_sum += reward * self.gamma ** (num_steps_taken - 1)

            if max_steps and num_steps_taken >= max_steps:
                break

        self.reset()
        return reward_sum, discounted_reward_sum

    def _process_state(self, raw_state: np.ndarray) -> Dict:
        processed_state = {}
        for i in range(self.state_dim):
            processed_state[i] = raw_state[i]
        return processed_state

    def sample_policy(self, state, use_continuous_action: bool, epsilon: float = 0.0):
        """
        Sample a random action
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        raw_action = self.env.action_space.sample()

        if self.action_type == EnvType.DISCRETE_ACTION:
            action_probability = 1.0 / self.action_dim
            if not use_continuous_action:
                return raw_action, str(self.state_dim + raw_action), action_probability
            action_vec = {self.state_dim + raw_action: 1}
            return raw_action, action_vec, action_probability

        if self.action_type == EnvType.CONTINUOUS_ACTION:
            # action_probability is the probability density of multi-variate
            # uniform distribution
            range_each_dim = (
                self.env.observation_space.high - self.env.observation_space.low
            )
            action_probability = 1.0 / reduce((lambda x, y: x * y), range_each_dim)
            action_vec = {}
            for i in range(self.action_dim):
                action_vec[self.state_dim + i] = raw_action[i]
            return raw_action, action_vec, action_probability

    def possible_actions(
        self,
        state,
        terminal: bool = False,
        ignore_terminal=False,
        use_continuous_action: bool = False,
        **kwargs,
    ):
        # possible actions will not be used in algorithms dealing with
        # continuous actions, so just return an empty list
        if terminal or self.action_type == EnvType.CONTINUOUS_ACTION:
            return []
        if not use_continuous_action:
            return self.actions
        possible_actions = []
        for i in range(self.action_dim):
            action_vec = {self.state_dim + i: 1}
            possible_actions.append(action_vec)
        return possible_actions
