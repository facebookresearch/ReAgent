#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# We removed Tensorflow dependencies.
# OutOfGraphReplayBuffer is renamed ReplayBuffer

# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The standard DQN replay memory.
This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""

import collections
import gzip
import logging
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)

try:
    import gym
    from gym import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    logger.warning(
        f"ReplayBuffer.create_from_env() will not work because gym is not installed"
    )

try:
    from recsim.simulator.recsim_gym import RecSimGymEnv

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False
    logger.warning(f"ReplayBuffer.create_from_env() will not recognize RecSim env")


# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = collections.namedtuple("shape_type", ["name", "shape", "type"])

# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = "$store$_"

# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4


def invalid_range(
    cursor: int, replay_capacity: int, stack_size: int, update_horizon: int
) -> np.ndarray:
    """Returns a array with the indices of cursor-related invalid transitions.
    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.
    It handles special cases in a circular buffer in the beginning and the end.
    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [
            (cursor - update_horizon + i) % replay_capacity
            for i in range(stack_size + update_horizon)
        ]
    )


class ReplayBuffer(object):
    """A simple Replay Buffer.
    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.
    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.
    Attributes:
      add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        stack_size: int,
        replay_capacity: int,
        batch_size: int,
        return_everything_as_stack: bool = False,
        update_horizon: int = 1,
        gamma: float = 0.99,
        max_sample_attempts: int = 1000,
        extra_storage_types: Optional[List[ReplayElement]] = None,
        observation_dtype=np.uint8,
        terminal_dtype=np.uint8,
        action_shape: Tuple[int, ...] = (),
        action_dtype=np.int32,
        reward_shape: Tuple[int, ...] = (),
        reward_dtype=np.float32,
    ) -> None:
        """Initializes ReplayBuffer.
        Args:
          observation_shape: tuple of ints.
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          return_everything_as_stack: bool, set True if we want everything,
             not just states, to be stacked too
          update_horizon: int, length of update ('n' in n-step update).
          gamma: int, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
          observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
          terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
            Atari 2600.
          action_shape: tuple of ints, the shape for the action vector. Empty tuple
            means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
            means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.
        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """
        assert isinstance(observation_shape, tuple)
        if replay_capacity < update_horizon + stack_size:
            raise ValueError(
                "There is not enough capacity to cover "
                "update_horizon and stack_size."
            )

        logger.info(
            "Creating a %s replay memory with the following parameters:",
            self.__class__.__name__,
        )
        logger.info("\t observation_shape: %s", str(observation_shape))
        logger.info("\t observation_dtype: %s", str(observation_dtype))
        logger.info("\t terminal_dtype: %s", str(terminal_dtype))
        logger.info("\t stack_size: %d", stack_size)
        logger.info("\t replay_capacity: %d", replay_capacity)
        logger.info("\t batch_size: %d", batch_size)
        logger.info("\t update_horizon: %d", update_horizon)
        logger.info("\t gamma: %f", gamma)

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._return_everything_as_stack = return_everything_as_stack
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        self.invalid_range = np.zeros((self._stack_size))
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)], dtype=np.float32
        )
        # track if index is valid for sampling purposes. there're two cases
        # 1) first stack_size-1 zero transitions at start of episode
        # 2) last update_horizon transitions before the cursor
        self._is_index_valid = np.zeros(self._replay_capacity, dtype=np.bool)
        self._num_valid_indices = 0
        self._num_transitions_in_current_episode = 0
        self._batch_type = collections.namedtuple(
            "batch_type", [e.name for e in self.get_transition_elements()]
        )

    @property
    def size(self) -> int:
        return self._num_valid_indices

    @classmethod
    def create_from_env(
        cls,
        env: "gym.Env",
        *,
        replay_memory_size: int,
        batch_size: int,
        stack_size: int = 1,
        store_log_prob: bool = True,
        **kwargs,
    ):
        extra_storage_types: List[ReplayElement] = []
        obs_space = env.observation_space

        if HAS_RECSIM and isinstance(env.unwrapped, RecSimGymEnv):
            assert isinstance(obs_space, spaces.Dict)
            user_obs_space = obs_space["user"]
            if not isinstance(user_obs_space, spaces.Box):
                raise NotImplementedError(
                    f"User observation space {type(user_obs_space)} is not supported"
                )
            # Put user into observation part of replay buffer
            observation_shape = user_obs_space.shape
            observation_dtype = user_obs_space.dtype

            # Create an element for doc & response
            extra_storage_types.extend(cls._get_replay_elements_for_recsim(obs_space))
        elif isinstance(obs_space, spaces.Box):
            observation_shape = obs_space.shape
            observation_dtype = obs_space.dtype
        else:
            raise NotImplementedError(
                f"Observation type {type(env.observation_space)} is not supported"
            )

        action_space = env.action_space
        if isinstance(action_space, (spaces.Box, spaces.MultiDiscrete)):
            action_dtype = action_space.dtype
            action_shape = action_space.shape
        elif isinstance(action_space, spaces.Discrete):
            action_dtype = action_space.dtype
            action_shape = ()
        else:
            raise NotImplementedError(
                f"env.action_space {type(env.action_space)} not supported."
            )

        extra_storage_types.append(ReplayElement("mdp_id", (), np.int64))
        extra_storage_types.append(ReplayElement("sequence_number", (), np.int64))
        if store_log_prob:
            extra_storage_types.append(ReplayElement("log_prob", (), np.float32))

        return cls(
            stack_size=stack_size,
            replay_capacity=replay_memory_size,
            batch_size=batch_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=(),
            reward_dtype=np.float32,
            extra_storage_types=extra_storage_types,
            **kwargs,
        )

    @staticmethod
    def _get_replay_elements_for_recsim(obs_space) -> List[ReplayElement]:
        """
        obs_space["doc"] is a dict with as many keys as number of candidates.
        All the values should be identical. They should be dict with keys
        corresponding to document features.

        obs_space["response"] is a tuple. Its length is the slate size presented
        to the user. Each element should be identical. They should be dict with
        keys corresponding to the type of response.
        """
        logger.info(obs_space)
        doc_obs_space = obs_space["doc"]
        if not isinstance(doc_obs_space, spaces.Dict):
            raise NotImplementedError(
                f"Doc space {type(doc_obs_space)} is not supported"
            )

        num_docs = len(doc_obs_space.spaces)

        # Assume that all docs are in the same space

        replay_elements: List[ReplayElement] = []

        doc_0_space = doc_obs_space["0"]
        if isinstance(doc_0_space, spaces.Dict):
            for k, v in doc_0_space.spaces.items():
                if isinstance(v, spaces.Discrete):
                    shape = (num_docs,)
                elif isinstance(v, spaces.Box):
                    shape = (num_docs, *v.shape)
                else:
                    raise NotImplementedError(
                        f"Doc feature {k} with the observation space of {type(v)}"
                        " is not supported"
                    )
                replay_elements.append(ReplayElement(f"doc_{k}", shape, v.dtype))
        elif isinstance(doc_0_space, spaces.Box):
            shape = (num_docs, *doc_0_space.shape)
            replay_elements.append(ReplayElement("doc", shape, doc_0_space.dtype))
        else:
            raise NotImplementedError(f"Unknown space: {doc_0_space}")

        augmentation = obs_space.spaces.get("augmentation", None)
        if augmentation is not None:
            aug_0_space = list(augmentation.spaces.values())[0]
            for k, v in aug_0_space.spaces.items():
                if isinstance(v, spaces.Discrete):
                    shape = (num_docs,)
                elif isinstance(v, spaces.Box):
                    shape = (num_docs, *v.shape)
                else:
                    raise NotImplementedError(
                        f"Augmentation feature {k} with the observation space "
                        f"of {type(v)} is not supported"
                    )
                replay_elements.append(
                    ReplayElement(f"augmentation_{k}", shape, v.dtype)
                )

        response_space = obs_space["response"]
        assert isinstance(response_space, spaces.Tuple)

        slate_size = len(response_space)

        response_space_0 = response_space[0]
        assert isinstance(response_space_0, spaces.Dict)
        for k, v in response_space_0.spaces.items():
            if isinstance(v, spaces.Discrete):
                shape = (slate_size,)
            elif isinstance(v, spaces.Box):
                shape = (slate_size, *v.shape)
            else:
                raise NotImplementedError(
                    f"Response {k} with the observation space of {type(v)} "
                    "is not supported"
                )
            replay_elements.append(ReplayElement(f"response_{k}", shape, v.dtype))

        return replay_elements

    def set_index_valid_status(self, idx: int, is_valid: bool):
        old_valid = self._is_index_valid[idx]
        if not old_valid and is_valid:
            self._num_valid_indices += 1
        elif old_valid and not is_valid:
            self._num_valid_indices -= 1
        assert self._num_valid_indices >= 0, f"{self._num_valid_indices} is negative"

        self._is_index_valid[idx] = is_valid

    def _create_storage(self) -> None:
        """Creates the numpy arrays used to store transitions.
        """
        self._store: Dict[str, np.ndarray] = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(
                array_shape, dtype=storage_element.type
            )

    def get_add_args_signature(self) -> List[ReplayElement]:
        """The signature of the add function.
        Note - Derived classes may return a different signature.
        Returns:
          list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        return self.get_storage_signature()

    def get_storage_signature(self) -> List[ReplayElement]:
        """Returns a default list of elements to be stored in this replay memory.
        Note - Derived classes may return a different signature.
        Returns:
          list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement(
                "observation", self._observation_shape, self._observation_dtype
            ),
            ReplayElement("action", self._action_shape, self._action_dtype),
            ReplayElement("reward", self._reward_shape, self._reward_dtype),
            ReplayElement("terminal", (), self._terminal_dtype),
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def _add_zero_transition(self) -> None:
        """Adds a padding transition filled with zeros (Used in episode beginnings).
        """
        zero_transition = []
        for element_type in self.get_add_args_signature():
            zero_transition.append(
                np.zeros(element_type.shape, dtype=element_type.type)
            )
        self._add(*zero_transition)

    def add(self, observation, action, reward, terminal, *args, **kwargs):
        """Adds a transition to the replay memory.
        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.
        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.
        If the replay memory is at capacity the oldest transition will be discarded.
        Args:
          observation: np.array with shape observation_shape.
          action: int, the action in the transition.
          reward: float, the reward received in the transition.
          terminal: np.dtype, acts as a boolean indicating whether the transition
                    was terminal (1) or not (0).
          *args: extra contents with shapes and dtypes according to
            extra_storage_types.
        """
        self._check_add_types(observation, action, reward, terminal, *args, **kwargs)
        last_idx = (self.cursor() - 1) % self._replay_capacity
        if self.is_empty() or self._store["terminal"][last_idx] == 1:
            self._num_transitions_in_current_episode = 0
            for _ in range(self._stack_size - 1):
                # Child classes can rely on the padding transitions being filled with
                # zeros. This is useful when there is a priority argument.
                self._add_zero_transition()

        # remember, the last update_horizon transitions shouldn't be sampled
        cur_idx = self.cursor()
        self.set_index_valid_status(idx=cur_idx, is_valid=False)
        if self._num_transitions_in_current_episode >= self._update_horizon:
            idx = (cur_idx - self._update_horizon) % self._replay_capacity
            self.set_index_valid_status(idx=idx, is_valid=True)
        self._add(observation, action, reward, terminal, *args, **kwargs)
        self._num_transitions_in_current_episode += 1

        # mark the next stack_size-1 as invalid (note cursor has advanced by 1)
        for i in range(self._stack_size - 1):
            idx = (self.cursor() + i) % self._replay_capacity
            self.set_index_valid_status(idx=idx, is_valid=False)

        if terminal:
            # Since the frame (cur_idx) we just inserted was terminal, we now mark
            # the last "num_back" transitions as valid for sampling (including cur_idx).
            # This is because next_state is not relevant for those terminal (multi-step)
            # transitions.
            # NOTE: this was not accounted for by the original Dopamine buffer.
            # It is not a big problem, since after update_horizon steps,
            # the original Dopamine buffer will make these frames
            # available for sampling.
            # But that is update_horizon steps too late. If we train right
            # after an episode terminates, this can result in missing the
            # bulk of rewards at the end of the most recent episode.
            num_back = min(
                self._num_transitions_in_current_episode, self._update_horizon
            )
            for i in range(0, num_back):
                idx = (cur_idx - i) % self._replay_capacity
                self.set_index_valid_status(idx=idx, is_valid=True)

    def _add(self, *args, **kwargs):
        """Internal add method to add to the storage arrays.
        Args:
          *args: All the elements in a transition.
        """
        self._check_args_length(*args, **kwargs)
        kwargs.update(
            {e.name: arg for arg, e in zip(args, self.get_add_args_signature())}
        )
        self._add_transition(kwargs)

    def _add_transition(self, transition):
        """Internal add method to add transition dictionary to storage arrays.
        Args:
          transition: The dictionary of names and values of the transition
                      to add to the storage.
        """
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1
        self.invalid_range = invalid_range(
            self.cursor(), self._replay_capacity, self._stack_size, self._update_horizon
        )

    def _check_args_length(self, *args, **kwargs):
        """Check if args passed to the add method have the same length as storage.
        Args:
          *args: Args for elements used in storage.
        Raises:
          ValueError: If args have wrong length.
        """
        if len(args) + len(kwargs) != len(self.get_add_args_signature()):
            raise ValueError(
                f"Add expects: {self.get_add_args_signature()}; "
                f" received {args} {kwargs}"
            )

    def _check_add_types(self, *args, **kwargs):
        """Checks if args passed to the add method match those of the storage.
        Args:
          *args: Args whose types need to be validated.
        Raises:
          ValueError: If args have wrong shape or dtype.
        """
        self._check_args_length(*args, **kwargs)
        add_arg_signature = self.get_add_args_signature()

        def _check(arg_element, store_element):
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO(b/80536437). This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError(
                    "arg {} has shape {}, expected {}".format(
                        store_element.name, arg_shape, store_element_shape
                    )
                )

        for arg_element, store_element in zip(args, add_arg_signature):
            _check(arg_element, store_element)

        for store_element in add_arg_signature[len(args) :]:
            arg_element = kwargs[store_element.name]
            _check(arg_element, store_element)

    def is_empty(self) -> bool:
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self) -> bool:
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_capacity

    def cursor(self) -> int:
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_capacity

    def get_range(
        self, array: np.ndarray, start_index: int, end_index: int
    ) -> np.ndarray:
        """Returns the range of array at the index handling wraparound if necessary.
        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.
        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, "end_index must be larger than start_index"
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), "Index {} has not been added.".format(
                start_index
            )

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = array[start_index:end_index, ...]
        # Slow list read.
        else:
            indices = [
                (start_index + i) % self._replay_capacity
                for i in range(end_index - start_index)
            ]
            return_array = array[indices, ...]
        return return_array

    def get_observation_stack(self, index):
        return self._get_element_stack(index, "observation")

    def _get_element_stack(self, index, element_name):
        state = self.get_range(
            self._store[element_name], index - self._stack_size + 1, index + 1
        )
        # The stacking axis is 0 but the agent expects as the last axis.
        return np.moveaxis(state, 0, -1)

    def get_terminal_stack(self, index):
        return self.get_range(
            self._store["terminal"], index - self._stack_size + 1, index + 1
        )

    def is_valid_transition(self, index):
        return self._is_index_valid[index]

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.
        Args:
          batch_size: int, number of indices returned.
        Returns:
          list of ints, a batch of valid indices sampled uniformly.
        Raises:
          RuntimeError: If there are no valid indices to sample.
        """
        if self._num_valid_indices == 0:
            raise RuntimeError(
                f"Cannot sample {batch_size} since there are no valid indices so far."
            )
        p = self._is_index_valid.astype(np.float64) / float(self._num_valid_indices)
        indices = np.random.choice(
            a=self._replay_capacity, size=batch_size, replace=True, p=p
        )
        return indices

    def sample_transition_batch_tensor(self, batch_size=None, indices=None):
        """
        Like sample_transition_batch, but returns torch tensors. Also, reshaping to
        our common shapes. I.e., state is 2-D unless stack_size > 1.
        Scalar values are returned as (batch_size, 1) instead of (batch_size,)
        """
        batch = self.sample_transition_batch(batch_size=batch_size, indices=indices)

        def _normalize_tensor(k, v):
            squeeze_set = {"state", "next_state"}
            t = torch.tensor(v)
            if k in squeeze_set and self._stack_size == 1:
                t = t.squeeze(2)
            elif t.ndim == 1:
                t = t.unsqueeze(1)
            return t

        return batch._replace(
            **{k: _normalize_tensor(k, v) for k, v in batch._asdict().items()}
        )

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).
        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        PrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.
        When the transition is terminal next_state_batch has undefined contents.
        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        assert isinstance(
            indices, np.ndarray
        ), f"Indices {indices} have type {type(indices)} instead of np.darray"
        assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)

        def get_stack_for_indices(key, indices):
            """ Get stack of observations """
            # calculate 2d array of indices with size (batch_size, stack_size)
            # ith row contain indices in the stack of obs at indices[i]
            stack_indices = indices.reshape(-1, 1) + np.arange(-self._stack_size + 1, 1)
            stack_indices %= self._replay_capacity
            retval = self._store[key][stack_indices]
            if len(retval.shape) > 2:
                # Reshape to (batch_size, obs_shape, stack_size)
                perm = [0] + list(range(2, len(self._observation_shape) + 2)) + [1]
                retval = retval.transpose(perm)
            return retval

        # calculate 2d array of indices with size (batch_size, update_horizon)
        # ith row contain the multistep indices starting at indices[i]
        multistep_indices = indices.reshape(-1, 1) + np.arange(self._update_horizon)
        multistep_indices %= self._replay_capacity

        def get_traj_lengths():
            """ Calculate trajectory length, defined to be the number of states
            in this multi_step transition until terminal state or until
            end of multi_step (a.k.a. update_horizon).
            """
            terminals = self._store["terminal"][multistep_indices]
            # if trajectory is non-terminal, we'll have traj_length = update_horizon
            terminals[:, -1] = True
            # Argmax find the first True in each one
            traj_lengths = np.argmax(terminals.astype(np.bool), axis=1) + 1
            return traj_lengths

        traj_lengths = get_traj_lengths()
        next_indices = (indices + traj_lengths) % self._replay_capacity

        def get_multistep_reward_for_indices():
            """ Sums up the reward for trajectory. """
            decays = self._gamma ** np.arange(self._update_horizon)
            decays = decays.reshape(1, self._update_horizon)
            masks = np.arange(self._update_horizon) < traj_lengths.reshape(-1, 1)
            rewards = self._store["reward"][multistep_indices] * decays * masks
            return rewards.sum(axis=1)

        batch_arrays = []
        for element in transition_elements:
            if element.name == "state":
                batch = get_stack_for_indices("observation", indices)
            elif element.name == "next_state":
                batch = get_stack_for_indices("observation", next_indices)
            elif element.name == "reward":
                if self._return_everything_as_stack:
                    if self._update_horizon > 1:
                        raise NotImplementedError(
                            "Uncertain how to do this without double counting.."
                        )
                    batch = get_stack_for_indices("reward", indices)
                else:
                    batch = get_multistep_reward_for_indices()
            elif element.name == "terminal":
                terminal_indices = (next_indices - 1) % self._replay_capacity
                if self._return_everything_as_stack:
                    batch = get_stack_for_indices("terminal", terminal_indices)
                else:
                    batch = self._store["terminal"][terminal_indices]
                batch = batch.astype(np.bool)
            elif element.name == "indices":
                batch = indices
            elif element.name in self._store:
                if self._return_everything_as_stack:
                    batch = get_stack_for_indices(element.name, indices)
                else:
                    batch = self._store[element.name][indices]
            elif element.name.startswith("next_"):
                store_name = element.name[len("next_") :]
                assert (
                    store_name in self._store
                ), f"{store_name} is not in {self._store.keys()}"
                if self._return_everything_as_stack:
                    batch = get_stack_for_indices(store_name, next_indices)
                else:
                    batch = self._store[store_name][next_indices]

            batch = batch.astype(element.type)
            batch_arrays.append(batch)

        batch_arrays = self._batch_type(*batch_arrays)

        # We assume the other elements are filled in by the subclass.
        return batch_arrays

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement(
                "state", (batch_size,) + self._state_shape, self._observation_dtype
            ),
            ReplayElement(
                "action", (batch_size,) + self._action_shape, self._action_dtype
            ),
            ReplayElement(
                "reward", (batch_size,) + self._reward_shape, self._reward_dtype
            ),
            ReplayElement(
                "next_state", (batch_size,) + self._state_shape, self._observation_dtype
            ),
            ReplayElement(
                "next_action", (batch_size,) + self._action_shape, self._action_dtype
            ),
            ReplayElement(
                "next_reward", (batch_size,) + self._reward_shape, self._reward_dtype
            ),
            ReplayElement("terminal", (batch_size,), self._terminal_dtype),
            ReplayElement("indices", (batch_size,), np.int32),
        ]
        for element in self._extra_storage_types:
            for prefix in ["", "next_"]:
                transition_elements.append(
                    ReplayElement(
                        f"{prefix}{element.name}",
                        (batch_size,) + tuple(element.shape),
                        element.type,
                    )
                )
        return transition_elements

    def _generate_filename(self, checkpoint_dir, name, suffix):
        return os.path.join(checkpoint_dir, "{}_ckpt.{}.gz".format(name, suffix))

    def _return_checkpointable_elements(self):
        """Return the dict of elements of the class for checkpointing.
        Returns:
          checkpointable_elements: dict containing all non private (starting with
          _) members + all the arrays inside self._store.
        """
        checkpointable_elements = {}
        for member_name, member in self.__dict__.items():
            if member_name == "_store":
                for array_name, array in self._store.items():
                    checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
            elif not member_name.startswith("_"):
                checkpointable_elements[member_name] = member
        return checkpointable_elements

    def save(self, checkpoint_dir, iteration_number):
        """Save the ReplayBuffer attributes into a file.
        This method will save all the replay buffer's state in a single file.
        Args:
          checkpoint_dir: str, the directory where numpy checkpoint files should be
            saved.
          iteration_number: int, iteration_number to use as a suffix in naming
            numpy checkpoint files.
        """
        if not os.path.exists(checkpoint_dir):
            return

        checkpointable_elements = self._return_checkpointable_elements()

        for attr in checkpointable_elements:
            filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    # Checkpoint the np arrays in self._store with np.save instead of
                    # pickling the dictionary is critical for file size and performance.
                    # STORE_FILENAME_PREFIX indicates that the variable is contained in
                    # self._store.
                    if attr.startswith(STORE_FILENAME_PREFIX):
                        array_name = attr[len(STORE_FILENAME_PREFIX) :]
                        np.save(outfile, self._store[array_name], allow_pickle=False)
                    # Some numpy arrays might not be part of storage
                    elif isinstance(self.__dict__[attr], np.ndarray):
                        np.save(outfile, self.__dict__[attr], allow_pickle=False)
                    else:
                        pickle.dump(self.__dict__[attr], outfile)

            # After writing a checkpoint file, we garbage collect the checkpoint file
            # that is four versions old.
            stale_iteration_number = iteration_number - CHECKPOINT_DURATION
            if stale_iteration_number >= 0:
                stale_filename = self._generate_filename(
                    checkpoint_dir, attr, stale_iteration_number
                )
                try:
                    os.remove(stale_filename)
                except FileNotFoundError:
                    pass

    def load(self, checkpoint_dir, suffix):
        """Restores the object from bundle_dictionary and numpy checkpoints.
        Args:
          checkpoint_dir: str, the directory where to read the numpy checkpointed
            files from.
          suffix: str, the suffix to use in numpy checkpoint files.
        Raises:
          NotFoundError: If not all expected files are found in directory.
        """
        save_elements = self._return_checkpointable_elements()
        # We will first make sure we have all the necessary files available to avoid
        # loading a partially-specified (i.e. corrupted) replay buffer.
        for attr in save_elements:
            filename = self._generate_filename(checkpoint_dir, attr, suffix)
            if not os.path.exists(filename):
                raise FileNotFoundError(None, None, "Missing file: {}".format(filename))
        # If we've reached this point then we have verified that all expected files
        # are available.
        for attr in save_elements:
            filename = self._generate_filename(checkpoint_dir, attr, suffix)
            with open(filename, "rb") as f:
                with gzip.GzipFile(fileobj=f) as infile:
                    if attr.startswith(STORE_FILENAME_PREFIX):
                        array_name = attr[len(STORE_FILENAME_PREFIX) :]
                        self._store[array_name] = np.load(infile, allow_pickle=False)
                    elif isinstance(self.__dict__[attr], np.ndarray):
                        self.__dict__[attr] = np.load(infile, allow_pickle=False)
                    else:
                        self.__dict__[attr] = pickle.load(infile)
