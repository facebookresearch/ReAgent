#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import enum
import logging
from functools import reduce
from typing import Dict, Optional, Tuple, Union

import gym
import numpy as np
import reagent.gym.envs  # noqa
import torch
from gym import Env
from reagent.gym.envs.env_factory import EnvFactory
from reagent.test.base.utils import only_continuous_normalizer
from reagent.test.environment.environment import Environment
from reagent.training.on_policy_predictor import OnPolicyPredictor


logger = logging.getLogger(__name__)


class ModelType(enum.Enum):
    CONTINUOUS_ACTION = "continuous"
    SOFT_ACTOR_CRITIC = "soft_actor_critic"
    TD3 = "td3"
    PYTORCH_DISCRETE_DQN = "pytorch_discrete_dqn"
    PYTORCH_PARAMETRIC_DQN = "pytorch_parametric_dqn"
    CEM = "cross_entropy_method"


class EnvType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    CONTINUOUS_ACTION = "continuous"
    UNKNOWN = "unknown"


class OpenAIGymEnvironment(Environment):
    def __init__(
        self,
        gymenv: Union[str, Env],
        epsilon=0,
        softmax_policy=False,
        gamma=0.99,
        epsilon_decay=1,
        minimum_epsilon=None,
        random_seed: Optional[int] = None,
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
        :param random_seed: The random seed for the environment
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.softmax_policy = softmax_policy
        self.gamma = gamma
        self.action_type: EnvType = EnvType.UNKNOWN
        self.state_dim = -1
        self.action_dim = -1
        self._create_env(gymenv, random_seed)
        assert self.action_type is not EnvType.UNKNOWN
        # allow only one type of exploration, softmax or epsilon-greedy
        assert not softmax_policy or epsilon == 0

        # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `img`.
        if not self.img:
            assert self.state_dim > 0
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        if self.action_type == EnvType.DISCRETE_ACTION:
            assert self.action_dim > 0
            self.actions = [str(a + self.state_dim) for a in range(self.action_dim)]

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)

    def _create_env(self, gymenv: Union[str, Env], random_seed: Optional[int]):
        """
        Creates a gym environment object and checks if it is supported. We
        support environments that supply Box(x, ) state representations and
        require Discrete(y) or Box(y,) action inputs.

        :param gymenv: String identifier for desired environment or environment
            object itself.
        """
        if isinstance(gymenv, Env):
            # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `env`.
            self.env = gymenv
            # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `env_name`.
            self.env_name = gymenv.unwrapped.spec.id
        else:
            if gymenv not in [e.id for e in gym.envs.registry.all()]:
                raise Exception("Env {} not found in OpenAI Gym.".format(gymenv))
            self.env = EnvFactory.make(gymenv)
            self.env_name = gymenv
            if random_seed is not None:
                self.env.seed(random_seed)
                self.env.action_space.seed(random_seed)

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
            # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `img`.
            self.img = False
        elif len(self.env.observation_space.shape) == 3:
            (
                # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `height`.
                self.height,
                # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `width`.
                self.width,
                # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute
                #  `num_input_channels`.
                self.num_input_channels,
            ) = self.env.observation_space.shape
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
        self, predictor: Optional[OnPolicyPredictor], state, test
    ) -> Tuple[torch.Tensor, float]:
        """
        Selects the next action.

        :param predictor: OnPolicyPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param state: State to evaluate predictor's policy on.
        :param test: Whether or not to bypass exploration (if predictor is not None).
            For discrete action problems, the exploration policy is epsilon-greedy
            or softmax.
            For continuous action problems, the exploration is achieved by adding
            noise to action outputs.

        Return an action vector in torch.Tensor format and action probability
        """
        # policy() is applied on a single state at a time
        assert len(state.size()) == 1

        # Convert state to batch of size 1
        state = state.unsqueeze(0)

        if predictor is None:
            # if no predictor is provided, random sample an action
            raw_action, _, action_probability = self.sample_policy(
                state=None, use_continuous_action=False
            )
            if self.action_type == EnvType.DISCRETE_ACTION:
                action = torch.zeros([self.action_dim])
                action[raw_action] = 1.0
                return action, action_probability
            return torch.tensor(raw_action), action_probability

        action = torch.zeros([self.action_dim])

        if predictor.policy_net():
            # continuous action space, policy network
            assert self.action_type == EnvType.CONTINUOUS_ACTION
            # pyre-fixme[16]: `OnPolicyPredictor` has no attribute `policy`.
            action_set = predictor.policy(state)
            action, action_probability = action_set.greedy, action_set.greedy_propensity
            action = action[0, :]
            return action, action_probability

        # Discrete action space
        assert self.action_type == EnvType.DISCRETE_ACTION

        if predictor.discrete_action():
            # DQN
            policy_action_set = predictor.policy(
                state, possible_actions_presence=torch.ones([1, self.action_dim])
            )
        else:
            # Parametric DQN, applied on discrete-action environments
            states_tiled = torch.repeat_interleave(
                state, repeats=self.action_dim, axis=0
            )
            policy_action_set = predictor.policy(
                states_tiled,
                (
                    torch.eye(self.action_dim),
                    torch.ones((self.action_dim, self.action_dim)),
                ),
            )
        if self.softmax_policy:
            action[policy_action_set.softmax] = 1.0
        else:
            action[policy_action_set.greedy] = 1.0

        if test:
            action_probability = 1.0
            return action, action_probability
        elif self.softmax_policy:
            action_probability = policy_action_set.softmax_act_prob
            return action, action_probability

        # epsilon-greedy
        action_probability = 1.0 - self.epsilon + self.epsilon / self.action_dim
        if float(torch.rand(1)) >= self.epsilon:
            return action, action_probability
        random_action, _, _ = self.sample_policy(
            state=None, use_continuous_action=False
        )
        random_action_tensor = torch.zeros([self.action_dim])
        random_action_tensor[random_action] = 1.0
        if torch.eq(random_action_tensor, action).all():
            return action, action_probability
        else:
            return random_action_tensor, self.epsilon / self.action_dim

    def run_ep_n_times(
        self,
        n,
        predictor: Optional[OnPolicyPredictor],
        max_steps=None,
        test=False,
        render=False,
    ):
        """
        Runs an episode of the environment n times and returns the average
        sum of rewards.

        :param n: Number of episodes to average over.
        :param predictor: OnPolicyPredictor object whose policy to
            follow. If set to None, use a random policy
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        reward_sum = 0.0
        discounted_reward_sum = 0.0
        for _ in range(n):
            ep_rew_sum, ep_raw_discounted_sum = self.run_episode(
                predictor, max_steps, test, render
            )
            reward_sum += ep_rew_sum
            discounted_reward_sum += ep_raw_discounted_sum
        avg_rewards = round(reward_sum / n, 2)
        avg_discounted_rewards = round(discounted_reward_sum / n, 2)
        return avg_rewards, avg_discounted_rewards

    def transform_state(self, state: np.ndarray) -> torch.Tensor:
        torch_state = torch.from_numpy(state).float()
        # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `img`.
        if self.img:
            # Convert from Height-Width-Channel into Channel-Height-Width
            torch_state = torch.transpose(torch_state, axes=[2, 0, 1])
        return torch_state

    def run_episode(
        self,
        predictor: Optional[OnPolicyPredictor],
        max_steps=None,
        test=False,
        render=False,
    ):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: OnPolicyPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        terminal = False
        next_state_numpy = self.reset()
        next_state = self.transform_state(next_state_numpy)
        next_action, _ = self.policy(predictor, next_state, test)
        reward_sum = 0.0
        discounted_reward_sum = 0
        num_steps_taken = 0

        while not terminal:
            logger.debug(
                f"OpenAIGym: {num_steps_taken}-th step, state: {next_state_numpy}"
            )
            action = next_action
            if render:
                # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `env`.
                self.env.render()

            if self.action_type == EnvType.DISCRETE_ACTION:
                action_index = int(torch.argmax(action))
                next_state_numpy, reward, terminal, _ = self.step(action_index)
                logger.debug(
                    f"OpenAIGym: take action {action_index}, reward: {reward}, terminal: {terminal}"
                )
            else:
                next_state_numpy, reward, terminal, _ = self.step(action.numpy())
                logger.debug(
                    f"OpenAIGym: take action {action.numpy()}, reward: {reward}, terminal: {terminal}"
                )

            next_state = self.transform_state(next_state_numpy)
            num_steps_taken += 1
            next_action, _ = self.policy(predictor, next_state, test)
            reward_sum += float(reward)
            discounted_reward_sum += reward * self.gamma ** (num_steps_taken - 1)

            if max_steps and num_steps_taken >= max_steps:
                break

        self.reset()
        return reward_sum, discounted_reward_sum

    # pyre-fixme[14]: `_process_state` overrides method defined in `Environment`
    #  inconsistently.
    def _process_state(self, raw_state: np.ndarray) -> Dict:
        processed_state = {}
        for i in range(self.state_dim):
            processed_state[i] = raw_state[i]
        return processed_state

    def sample_policy(self, state, use_continuous_action: bool, epsilon: float = 1.0):
        """
        Sample a random action
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        # TODO: support epsilon greedy
        assert epsilon == 1.0

        # pyre-fixme[16]: `OpenAIGymEnvironment` has no attribute `env`.
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
