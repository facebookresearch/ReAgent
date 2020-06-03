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
        return_as_timeline_format: bool = False,
        update_horizon: int = 1,
        gamma: float = 0.99,
        max_sample_attempts: int = 1000,
        extra_storage_types: Optional[List[ReplayElement]] = None,
        observation_dtype=np.uint8,
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
          return_as_timeline_format: bool, when set True, next_(states, actions, etc.)
            is returned list format, like the output of TimelineOperator
          update_horizon: int, length of update ('n' in n-step update).
          gamma: int, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
          observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
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

        if return_as_timeline_format:
            if update_horizon <= 1:
                logger.warn(
                    f"Pointless to set return_as_timeline_format when "
                    f"update_horizon ({update_horizon}) isn't > 1."
                    "But we'll support it anyways..."
                )

        logger.info(
            "Creating a %s replay memory with the following parameters:",
            self.__class__.__name__,
        )
        logger.info("\t observation_shape: %s", str(observation_shape))
        logger.info("\t observation_dtype: %s", str(observation_dtype))
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
        self._return_as_timeline_format = return_as_timeline_format
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        # FIXME: np.bool causes UBSAN error
        self._terminal_dtype = np.uint8
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._decays = (self._gamma ** torch.arange(self._update_horizon)).unsqueeze(0)
        # track if index is valid for sampling purposes. there're two cases
        # 1) first stack_size-1 zero transitions at start of episode
        # 2) last update_horizon transitions before the cursor
        self._is_index_valid = torch.zeros(self._replay_capacity, dtype=torch.bool)
        self._num_valid_indices = 0
        self._num_transitions_in_current_episode = 0
        self._batch_type = collections.namedtuple(
            "batch_type", [e.name for e in self.get_transition_elements()]
        )
        self._key_to_shape_map = {k.name: k.shape for k in self.get_storage_signature()}

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
        if isinstance(
            action_space, (spaces.Box, spaces.MultiDiscrete, spaces.Discrete)
        ):
            action_dtype = action_space.dtype
            action_shape = action_space.shape
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
        self._store: Dict[str, torch.Tensor] = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = torch.from_numpy(
                np.empty(array_shape, dtype=storage_element.type)
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
        if self.is_empty() or self._store["terminal"][last_idx]:
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
        elements = self.get_add_args_signature()
        # convert kwarg np.arrays to torch.tensors
        for element in elements[len(args) :]:
            if element.name in kwargs:
                kwargs[element.name] = torch.from_numpy(
                    np.array(kwargs[element.name], dtype=element.type)
                )
        # convert arg np.arrays to torch.tensors
        kwargs.update(
            {
                e.name: torch.from_numpy(np.array(arg, dtype=e.type))
                for arg, e in zip(args, elements[: len(args)])
            }
        )
        self._add_transition(kwargs)

    def _add_transition(self, transition: Dict[str, torch.Tensor]) -> None:
        """Internal add method to add transition dictionary to storage arrays.
        Args:
          transition: The dictionary of names and values of the transition
                      to add to the storage.
        """
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1

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
                arg_shape = ()
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

    def is_valid_transition(self, index):
        return self._is_index_valid[index]

    def sample_index_batch(self, batch_size: int) -> torch.Tensor:
        """Returns a batch of valid indices sampled uniformly.
        Args:
          batch_size: int, number of indices returned.
        Returns:
          1D tensor of ints, a batch of valid indices sampled uniformly.
        Raises:
          RuntimeError: If there are no valid indices to sample.
        """
        if self._num_valid_indices == 0:
            raise RuntimeError(
                f"Cannot sample {batch_size} since there are no valid indices so far."
            )
        valid_indices = self._is_index_valid.nonzero().squeeze(1)
        return valid_indices[torch.randint(valid_indices.shape[0], (batch_size,))]

    def sample_all_valid_transitions(self):
        valid_indices = self._is_index_valid.nonzero().squeeze(1)
        assert (
            valid_indices.ndim == 1
        ), f"Expecting 1D tensor since is_index_valid is 1D. Got {valid_indices}."
        return self.sample_transition_batch(
            batch_size=len(valid_indices), indices=valid_indices
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
        NOTE: Tensors are reshaped. I.e., state is 2-D unless stack_size > 1.
        Scalar values are returned as (batch_size, 1) instead of (batch_size,).
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or Tensor, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
          transition_batch: tuple of Tensors with the shape and type as in
            get_transition_elements().
        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        else:
            assert isinstance(
                indices, torch.Tensor
            ), f"Indices {indices} have type {type(indices)} instead of torch.Tensor"
            indices = indices.type(dtype=torch.int64)
        assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)

        # calculate 2d array of indices with size (batch_size, update_horizon)
        # ith row contain the multistep indices starting at indices[i]
        multistep_indices = indices.unsqueeze(1) + torch.arange(self._update_horizon)
        multistep_indices %= self._replay_capacity

        steps = self._get_steps(multistep_indices)

        # to pass in to next_features and reward to toggle whether to return
        # a list batch of length steps.
        if self._return_as_timeline_format:
            next_indices = (indices + 1) % self._replay_capacity
            steps_for_timeline_format = steps
        else:
            next_indices = (indices + steps) % self._replay_capacity
            steps_for_timeline_format = None

        batch_arrays = []
        for element in transition_elements:
            if element.name == "state":
                batch = self._get_batch_for_indices("observation", indices)
            elif element.name == "next_state":
                batch = self._get_batch_for_indices(
                    "observation", next_indices, steps_for_timeline_format
                )
            elif element.name == "indices":
                batch = indices
            elif element.name == "terminal":
                terminal_indices = (indices + steps - 1) % self._replay_capacity
                batch = self._store["terminal"][terminal_indices].to(torch.bool)
            elif element.name == "reward":
                if self._return_as_timeline_format or self._return_everything_as_stack:
                    batch = self._get_batch_for_indices(
                        "reward", indices, steps_for_timeline_format
                    )
                else:
                    batch = self._reduce_multi_step_reward(multistep_indices, steps)
            elif element.name == "step":
                batch = steps
            elif element.name in self._store:
                batch = self._get_batch_for_indices(element.name, indices)
            elif element.name.startswith("next_"):
                store_name = element.name[len("next_") :]
                assert (
                    store_name in self._store
                ), f"{store_name} is not in {self._store.keys()}"
                batch = self._get_batch_for_indices(
                    store_name, next_indices, steps_for_timeline_format
                )
            else:
                # We assume the other elements are filled in by the subclass.
                batch = torch.from_numpy(np.empty(element.shape, dtype=element.type))

            # always enables the batch_size dim
            if isinstance(batch, torch.Tensor) and batch.ndim == 1:
                batch = batch.unsqueeze(1)
            batch_arrays.append(batch)

        batch_arrays = self._batch_type(*batch_arrays)
        return batch_arrays

    def _get_batch_for_indices(
        self, key: str, indices: torch.Tensor, steps: Optional[torch.Tensor] = None
    ):
        """ Get batch for given key.
            There are two orthogonal special cases.
            - returning a stack of features:
                View this case as adding an extra "stack" dimension to feature,
                causing the shape to be (*feature.shape, stack_size)
            - returning next_features as a list (same as timeline output):
                This should only be on if update_horizon is > 1.
                If this is the case then we don't return a torch.Tensor,
                but instead return List[List[features]] where the ith
                element is torch.tensor([feat_{t+1}, ..., feat_{t+k}]);
                where k <= multi_steps could be strictly less if there's a
                terminal state.
                NOTE: this option is activated by using the optional steps parameter.

            Otherwise, we just return the indexed features in the replay buffer.
            In all of the cases, we assume indices is 1-dimensional.
        """
        assert len(indices.shape) == 1, f"{indices.shape} isn't 1-dimensional."
        if steps is not None:
            # for next state timeline format
            assert indices.shape == steps.shape, f"{indices.shape} != {steps.shape}"
            return [
                self._get_stack_for_indices(
                    key, torch.arange(start_idx, start_idx + step)
                )
                for start_idx, step in zip(indices.tolist(), steps.tolist())
            ]
        else:
            return self._get_stack_for_indices(key, indices)

    def _reduce_multi_step_reward(
        self, multistep_indices: torch.Tensor, steps: torch.Tensor
    ):
        # default behavior is to sum up multi_step reward
        masks = torch.arange(self._update_horizon) < steps.unsqueeze(1)
        rewards = self._store["reward"][multistep_indices] * self._decays * masks
        return rewards.sum(dim=1)

    def _get_stack_for_indices(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """ Get stack of transition data. """
        assert len(indices.shape) == 1, f"{indices.shape} not 1-dimensional"
        feature_shape = self._key_to_shape_map[key]
        # calculate 2d array of indices of shape (batch_size, stack_size)
        # ith row contain indices in the stack of obs at indices[i]
        stack_indices = indices.unsqueeze(1) + torch.arange(-self._stack_size + 1, 1)
        # pyre-fixme[16]: `Tensor` has no attribute `__imod__`.
        stack_indices %= self._replay_capacity
        retval = self._store[key][stack_indices]
        # retval has shape (batch_size, stack_size, obs_shape) right now, so
        # reshape to (batch_size, obs_shape, stack_size)
        perm = [0] + list(range(2, len(feature_shape) + 2)) + [1]
        retval = retval.permute(*perm)
        # squeeze the stack dim if it is 1
        if self._stack_size == 1:
            retval = retval.squeeze(len(perm) - 1)
        return retval

    def _get_steps(self, multistep_indices: torch.Tensor) -> torch.Tensor:
        """ Calculate trajectory length, defined to be the number of states
        in this multi_step transition until terminal state or until
        end of multi_step (a.k.a. update_horizon).
        """
        terminals = self._store["terminal"][multistep_indices].to(torch.bool)
        # if trajectory is non-terminal, we'll have traj_length = update_horizon
        terminals[:, -1] = True
        # use argmax to find the first True in each trajectory
        # NOTE: argmax may not contain the first occurrence of each maximal value found,
        # unless it is unique, so we need to make each boolean unique,
        # with the first occurance the largarst number
        terminals = terminals.float()
        unique_mask = torch.arange(terminals.shape[1] + 1, 1, -1)
        terminals = torch.einsum("ab,b->ab", (terminals, unique_mask))
        return torch.argmax(terminals, dim=1) + 1

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
            ReplayElement("step", (batch_size,), np.int32),
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
        # TODO: Save tensors to torch files.
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
                        np.save(
                            outfile, self._store[array_name].numpy(), allow_pickle=False
                        )
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
        # TODO: Load tensors from torch files.
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
                        self._store[array_name] = torch.from_numpy(
                            np.load(infile, allow_pickle=False)
                        )
                    elif isinstance(self.__dict__[attr], np.ndarray):
                        self.__dict__[attr] = np.load(infile, allow_pickle=False)
                    else:
                        self.__dict__[attr] = pickle.load(infile)
