#!/usr/bin/env python3


import enum

import gym
import numpy as np

from caffe2.python import workspace

from ml.rl.test.utils import default_normalizer
from ml.rl.training.ddpg_predictor import DDPGPredictor
from ml.rl.training.continuous_action_dqn_predictor import ContinuousActionDQNPredictor
from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor


class ModelType(enum.Enum):
    DISCRETE_ACTION = 'discrete'
    PARAMETRIC_ACTION = 'parametric'
    CONTINUOUS_ACTION = 'continuous'


class EnvType(enum.Enum):
    DISCRETE_ACTION = 'discrete'
    CONTINUOUS_ACTION = 'continuous'


class OpenAIGymEnvironment:
    def __init__(
        self,
        gymenv,
        epsilon=0.2,
        softmax_policy=1,
        max_replay_memory_size=10000
    ):
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

        supports_state = isinstance(
            self.env.observation_space, gym.spaces.Box
        ) and len(self.env.observation_space.shape) in [1, 3]
        supports_action =\
            type(self.env.action_space) in (gym.spaces.Discrete, gym.spaces.Box)

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

        :param batch_size: Number of sampled transitions to return.
        """
        cols = [[], [], [], [], [], [], [], []]
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        return [np.array(x) for x in cols]

    def sample_and_load_training_data_c2(
        self,
        num_samples,
        model_type,
        maxq_learning,
    ):
        """
        Loads and preprocesses shuffled, transformed transitions from
        replay memory into the training net.

        :param num_samples: Number of transitions to sample from replay memory.
        :param model_type: Model type (discrete, parametric).
        :param maxq_learning: Boolean indicating to use q-learning or sarsa.
        """
        states, actions, rewards, next_states, next_actions,\
            terminals, possible_next_actions,\
            possible_next_actions_lengths = self.sample_memories(num_samples)

        workspace.FeedBlob('states', np.array(states, dtype=np.float32))
        workspace.FeedBlob('actions', np.array(actions, dtype=np.float32))
        workspace.FeedBlob(
            'rewards',
            np.array(rewards, dtype=np.float32).reshape(-1, 1)
        )
        workspace.FeedBlob(
            'next_states', np.array(next_states, dtype=np.float32)
        )
        workspace.FeedBlob(
            'not_terminals',
            np.logical_not(terminals, dtype=np.bool).reshape(-1, 1)
        )

        # SARSA algorithm does not need possible next actions so return
        if not maxq_learning:
            workspace.FeedBlob(
                'next_actions', np.array(next_actions, dtype=np.float32)
            )
            return

        if model_type == ModelType.DISCRETE_ACTION.value:
            possible_next_actions = np.array(possible_next_actions, np.float32)
            workspace.FeedBlob('possible_next_actions', possible_next_actions)
            return

        pnas = []
        for pna_matrix in possible_next_actions:
            for row in pna_matrix:
                pnas.append(row)

        workspace.FeedBlob(
            'possible_next_actions_lengths',
            np.array(possible_next_actions_lengths, dtype=np.int32)
        )
        workspace.FeedBlob(
            'possible_next_actions', np.array(pnas, dtype=np.float32)
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
            [
                x for x in
                list(range(self.state_dim, self.state_dim + self.action_dim))
            ]
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
        if not test and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            if not self.img:
                # Convert next_state to a list[dict[int,float]]
                next_state_dict = [{}]
                for i in range(next_state.shape[1]):
                    next_state_dict[0][i] = next_state[0][i]
            if isinstance(predictor, DiscreteActionPredictor):
                if self.img:
                    if self.softmax_policy:
                        action_str = predictor.policy_image(next_state)[1]
                    else:
                        action_str = predictor.policy_image(next_state)[0]
                else:
                    if self.softmax_policy:
                        action_str = predictor.policy(next_state_dict)[1]
                    else:
                        action_str = predictor.policy(next_state_dict)[0]
                action_idx = self.actions.index(action_str.decode("utf-8"))
            elif isinstance(predictor, ContinuousActionDQNPredictor):
                normed_action_keys = sorted(self.normalization_action.keys())
                states, actions = [], []
                for action_key in normed_action_keys:
                    states.append(next_state_dict[0])
                    actions.append({action_key: 1})
                if self.softmax_policy:
                    action_idx = predictor.policy(states, actions)[1]
                else:
                    action_idx = predictor.policy(states, actions)[0]
                action_to_take = list(actions[action_idx].keys())
                action_idx = normed_action_keys.index(action_to_take[0])
            elif isinstance(predictor, DDPGPredictor):
                if test:
                    return predictor.predict_action(
                        next_state_dict, noisy=False
                    )[0]
                return predictor.predict_action(next_state_dict, noisy=True)[0]
        action[action_idx] = 1.0
        return action

    def insert_into_memory(
        self, state, action, reward, next_state, next_action, terminal,
        possible_next_actions, possible_next_actions_lengths
    ):
        """
        Inserts transition into replay memory in such a way that retrieving
        transitions uniformly at random will be equivalent to reservoir sampling.
        """
        item = (
            state, action, reward, next_state, next_action, terminal,
            possible_next_actions, possible_next_actions_lengths
        )

        if self.memory_num < self.max_replay_memory_size:
            self.replay_memory.append(item)
        elif self.memory_num >= self.skip_insert_until:
            p = float(self.max_replay_memory_size) / self.memory_num
            self.skip_insert_until += np.random.geometric(p)
            rand_index = np.random.randint(self.max_replay_memory_size)
            self.replay_memory[rand_index] = item
        self.memory_num += 1

    def run_ep_n_times(
        self, n, predictor, max_steps=None, test=False, render=False
    ):
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
