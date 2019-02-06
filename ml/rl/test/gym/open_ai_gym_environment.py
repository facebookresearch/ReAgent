#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import enum
from functools import partial, reduce
from typing import Deque, List, Optional, Tuple, Union

import gym
import numpy as np
from ml.rl.test.gridworld.gridworld_base import (
    ACTION,
    FEATURES,
    MultiStepSamples,
    Samples,
)
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymPredictor,
    GymSACPredictor,
)
from ml.rl.test.utils import default_normalizer, only_continuous_normalizer
from ml.rl.training._dqn_predictor import _DQNPredictor
from ml.rl.training._parametric_dqn_predictor import _ParametricDQNPredictor
from ml.rl.training.actor_predictor import ActorPredictor
from ml.rl.training.ddpg_predictor import DDPGPredictor
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.training.rl_predictor_pytorch import RLPredictor


class ModelType(enum.Enum):
    CONTINUOUS_ACTION = "continuous"
    SOFT_ACTOR_CRITIC = "soft_actor_critic"
    DDPG_ACTOR_CRITIC = "ddpg_actor_critic"
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
        self.env_name_str = gymenv

        self._create_env(gymenv)
        if not self.img:
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        if self.action_type == EnvType.DISCRETE_ACTION:
            self.actions = [str(a + self.state_dim) for a in range(self.action_dim)]

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
            raw_action, _, action_probability = self._random_sample_action(False)
            if self.action_type == EnvType.DISCRETE_ACTION:
                action = np.zeros([self.action_dim], dtype=np.float32)
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
        elif isinstance(predictor, GymDDPGPredictor):
            # FIXME: need to calculate action probability properly
            action_probability = 0.0
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            if test:
                return predictor.policy(next_state)[0], action_probability
            return (
                predictor.policy(next_state, add_action_noise=True)[0],
                action_probability,
            )
        elif isinstance(predictor, GymSACPredictor):
            # FIXME: need to calculate action probability properly
            # FIXME: also need to support adding noise on outputs when test is True
            action_probability = 0.0
            if state_preprocessor:
                next_state = state_preprocessor.forward(next_state)
            return predictor.policy(next_state)[0], action_probability
        elif isinstance(predictor, (DQNPredictor, _DQNPredictor)):
            action_probability = 1.0 if test else 1.0 - self.epsilon
            # Use DQNPredictor directly - useful to test caffe2 predictor
            # assumes state preprocessor already part of predictor net.
            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            q_values = predictor.predict(sparse_next_states)
            action_idx = int(max(q_values[0], key=q_values[0].get)) - self.state_dim
            action[action_idx] = 1.0
            return action, action_probability
        elif isinstance(predictor, (ParametricDQNPredictor, _ParametricDQNPredictor)):
            # Needs to get a list of candidate actions if actions are continuous
            if self.action_type == EnvType.CONTINUOUS_ACTION:
                raise NotImplementedError()
            action_probability = 1.0 if test else 1.0 - self.epsilon
            next_state = np.repeat(next_state, repeats=self.action_dim, axis=0)
            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            sparse_actions = [
                {str(i + self.state_dim): 1} for i in range(self.action_dim)
            ]
            q_values = predictor.predict(sparse_next_states, None, sparse_actions)
            q_values = np.fromiter(map(lambda x: x["Q"], q_values), np.float).reshape(
                self.action_dim
            )
            action_idx = np.argmax(q_values)
            action[action_idx] = 1.0
            return action, action_probability
        elif isinstance(predictor, (DDPGPredictor, ActorPredictor)):
            # FIXME need to calculate action probability properly
            action_probability = 0.0
            if not test and np.random.rand() < self.epsilon:
                # FIXME: get sac/ddpg output with noise
                raw_action = self.env.action_space.sample()
                if self.action_type == EnvType.DISCRETE_ACTION:
                    action[raw_action] = 1.0
                else:
                    action = raw_action
                return action, action_probability

            sparse_next_states = predictor.in_order_dense_to_sparse(next_state)
            prediction = predictor.actor_prediction(sparse_next_states)[0]
            if self.action_type == EnvType.DISCRETE_ACTION:
                raw_action = (
                    int(max(prediction, key=(lambda key: prediction[key])))
                    - self.state_dim
                )
                action[raw_action] = 1.0
            else:
                action[:] = [prediction[k] for k in sorted(prediction.keys())]
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

    def transform_state(self, state):
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

    def generate_random_samples(
        self,
        num_transitions: int,
        use_continuous_action: bool,
        multi_steps: Optional[int] = None,
    ) -> Union[Samples, MultiStepSamples]:
        # generate samples totally randomly
        assert self.epsilon == 1.0
        assert not self.img, "Not support for image-based environments for now"
        samples = self._generate_random_samples(
            num_transitions, use_continuous_action, multi_steps, 1000
        )
        return samples

    @staticmethod
    def set_if_in_range(index, limit, container, value):
        if index >= limit:
            return
        container[index] = value

    def _process_state(self, raw_state):
        processed_state = {}
        for i in range(self.state_dim):
            processed_state[i] = raw_state[i]
        return processed_state

    def _random_sample_action(self, use_continuous_action: bool):
        """
        Sample a random action by following epsilon-greedy
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        raw_action = self.env.action_space.sample()

        if self.action_type == EnvType.DISCRETE_ACTION:
            action_probability = self.epsilon / self.action_dim
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

    def _possible_actions(self, terminal: bool, use_continuous_action: bool):
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

    def _generate_random_samples(
        self,
        num_transitions: int,
        use_continuous_action: bool,
        multi_steps: Optional[int] = None,
        max_step: Optional[int] = None,
    ) -> Union[Samples, MultiStepSamples]:
        """ Generate samples:
            [
             s_t,
             (a_t, a_{t+1}, ..., a_{t+steps}),
             (r_t, r_{t+1}, ..., r_{t+steps}),
             (s_{t+1}, s_{t+2}, ..., s_{t+steps+1})
            ]

        :param num_transitions: How many transitions to collect
        :param use_continuous_action: True if action is represented as
            a vector using a dictionary; otherwise action is represented as string
        :param multi_steps: An integer, if provided, decides how many steps of
            transitions contained in each sample. Only used if you want to train
            multi-step RL.
        """
        return_single_step_samples = False
        if multi_steps is None:
            return_single_step_samples = True
            multi_steps = 1

        # Initialize lists
        states: List[FEATURES] = [{} for _ in range(num_transitions)]
        action_probabilities: List[float] = [0.0] * num_transitions
        rewards: List[List[float]] = [[] for _ in range(num_transitions)]
        next_states: List[List[FEATURES]] = [[{}] for _ in range(num_transitions)]
        terminals: List[List[bool]] = [[] for _ in range(num_transitions)]
        mdp_ids = [""] * num_transitions
        sequence_numbers = [0] * num_transitions
        possible_actions: List[List[ACTION]] = [[] for _ in range(num_transitions)]
        possible_next_actions: List[List[List[ACTION]]] = [
            [[]] for _ in range(num_transitions)
        ]
        if use_continuous_action:
            actions: List[FEATURES] = [{} for _ in range(num_transitions)]
            next_actions: List[List[FEATURES]] = [[] for _ in range(num_transitions)]
        else:
            actions: List[str] = [""] * num_transitions  # noqa
            next_actions: List[List[str]] = [[] for _ in range(num_transitions)]  # noqa

        state = None
        terminal = True
        next_raw_action = None
        next_processed_action = None
        next_action_probability = 1.0
        transition = 0
        mdp_id = -1
        sequence_number = 0

        state_deque: Deque[FEATURES] = collections.deque(maxlen=multi_steps)
        action_deque: Deque[ACTION] = collections.deque(maxlen=multi_steps)
        action_probability_deque: Deque[float] = collections.deque(maxlen=multi_steps)
        reward_deque: Deque[float] = collections.deque(maxlen=multi_steps)
        next_state_deque: Deque[FEATURES] = collections.deque(maxlen=multi_steps)
        next_action_deque: Deque[ACTION] = collections.deque(maxlen=multi_steps)
        terminal_deque: Deque[bool] = collections.deque(maxlen=multi_steps)
        sequence_number_deque: Deque[int] = collections.deque(maxlen=multi_steps)
        possible_action_deque: Deque[List[ACTION]] = collections.deque(
            maxlen=multi_steps
        )
        possible_next_action_deque: Deque[List[ACTION]] = collections.deque(
            maxlen=multi_steps
        )

        # We run until we finish the episode that completes N transitions, but
        # we may have to go beyond N to reach the end of that episode
        while not terminal or transition < num_transitions:
            if terminal:
                state = self.env.reset()
                terminal = False
                mdp_id += 1
                sequence_number = 0
                state_deque.clear()
                action_deque.clear()
                action_probability_deque.clear()
                reward_deque.clear()
                next_state_deque.clear()
                next_action_deque.clear()
                terminal_deque.clear()
                sequence_number_deque.clear()
                possible_action_deque.clear()
                possible_next_action_deque.clear()
                raw_action, processed_action, action_probability = self._random_sample_action(
                    use_continuous_action
                )
            else:
                raw_action = next_raw_action
                processed_action = next_processed_action
                action_probability = next_action_probability
                sequence_number += 1

            possible_action = self._possible_actions(terminal, use_continuous_action)

            next_state, reward, terminal, _ = self.env.step(raw_action)
            if max_step is not None and sequence_number >= max_step:
                terminal = True
            next_raw_action, next_processed_action, next_action_probability = self._random_sample_action(
                use_continuous_action
            )
            possible_next_action = self._possible_actions(
                terminal, use_continuous_action
            )

            state_deque.append(self._process_state(state))
            action_deque.append(processed_action)
            action_probability_deque.append(action_probability)
            reward_deque.append(reward)
            # Format terminals in same way we ask clients to log terminals (in RL dex)
            if terminal:
                next_processed_state = {}  # noqa
            else:
                next_processed_state = self._process_state(next_state)
            next_state_deque.append(next_processed_state)
            next_action_deque.append(next_processed_action)
            terminal_deque.append(terminal)
            sequence_number_deque.append(sequence_number)
            possible_action_deque.append(possible_action)
            possible_next_action_deque.append(possible_next_action)

            # We want exactly N data points, but we need to wait until the
            # episode is over so we can get the episode values. `set_if_in_range`
            # will set episode values if they are in the range [0,N) and ignore
            # otherwise.
            if not terminal and len(state_deque) == multi_steps:
                set_if_in_range = partial(
                    self.set_if_in_range, transition, num_transitions
                )
                set_if_in_range(states, state_deque[0])
                set_if_in_range(actions, action_deque[0])
                set_if_in_range(action_probabilities, action_probability_deque[0])
                set_if_in_range(rewards, list(reward_deque))
                set_if_in_range(next_states, list(next_state_deque))
                set_if_in_range(next_actions, list(next_action_deque))
                set_if_in_range(terminals, list(terminal_deque))
                set_if_in_range(mdp_ids, str(mdp_id))
                set_if_in_range(sequence_numbers, sequence_number_deque[0])
                set_if_in_range(possible_actions, possible_action_deque[0])
                set_if_in_range(possible_next_actions, list(possible_next_action_deque))
                transition += 1
            # collect samples at the end of the episode. The steps between state
            # and next_state can be less than or equal to `multi_steps`
            if terminal:
                for _ in range(len(state_deque)):
                    set_if_in_range = partial(
                        self.set_if_in_range, transition, num_transitions
                    )
                    set_if_in_range(states, state_deque.popleft())
                    set_if_in_range(actions, action_deque.popleft())
                    set_if_in_range(
                        action_probabilities, action_probability_deque.popleft()
                    )
                    set_if_in_range(rewards, list(reward_deque))
                    set_if_in_range(next_states, list(next_state_deque))
                    set_if_in_range(next_actions, list(next_action_deque))
                    set_if_in_range(terminals, list(terminal_deque))
                    set_if_in_range(mdp_ids, str(mdp_id))
                    set_if_in_range(sequence_numbers, sequence_number_deque.popleft())
                    set_if_in_range(possible_actions, possible_action_deque.popleft())
                    set_if_in_range(
                        possible_next_actions, list(possible_next_action_deque)
                    )
                    reward_deque.popleft()
                    next_state_deque.popleft()
                    next_action_deque.popleft()
                    terminal_deque.popleft()
                    possible_next_action_deque.popleft()
                    transition += 1

            state = next_state

        samples = MultiStepSamples(  # noqa
            mdp_ids=mdp_ids,
            sequence_numbers=sequence_numbers,
            states=states,
            actions=actions,
            action_probabilities=action_probabilities,
            rewards=rewards,
            possible_actions=possible_actions,
            next_states=next_states,
            next_actions=next_actions,
            terminals=terminals,
            possible_next_actions=possible_next_actions,
        )
        if return_single_step_samples:
            return samples.to_single_step()
        self.env.reset()
        return samples
