#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

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

import abc
import collections
import gzip
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class ElementMetadata:
    @classmethod
    @abc.abstractmethod
    def create_from_example(cls, name: str, example):
        """Constructor of the Metadata.
        Given an input example, construct an ElementMetadata for this key `name`.
        Good practice to call self.validate here after initializing metadata.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def zero_example(self):
        """What would an empty `input` example look like?"""
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, name: str, input):
        """Does the input look correct?"""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_storage(self, capacity: int):
        """Initialize the replay buffer with given `capacity`, for this data type.
        I.e. what is the "internal representation" of this data type in the replay buffer?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def input_to_storage(self, input):
        """Convert `input` to the "internal representation" of the replay buffer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_to_output(self, sample):
        """Convert "internal representation" of replay buffer to `output`.
        Concretely, when we call replay_buffer.sample(...), what do we want the output to look like?
        """
        raise NotImplementedError()


@dataclass
class DenseMetadata(ElementMetadata):
    """
    Internal representation is a torch tensor.
    Batched output is tensor of shape (batch_size, obs_shape, stack_size)
    """

    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def create_from_example(cls, name: str, example):
        arr = np.array(example)
        dtype = arr.dtype
        if dtype == np.dtype("float64"):
            dtype = np.dtype("float32")
        res = cls(arr.shape, dtype)
        res.validate(name, example)
        return res

    def zero_example(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def validate(self, name: str, input):
        assert not isinstance(input, (dict, torch.Tensor)), (
            f"{name}: {type(input)} is dict or torch.Tensor"
        )
        arr = np.array(input)
        dtype = arr.dtype
        if dtype == np.dtype("float64"):
            dtype = np.dtype("float32")
        assert arr.shape == self.shape and dtype == self.dtype, (
            f"{name}: Expected {self.shape} {self.dtype}, got {arr.shape} {dtype}"
        )

    def create_storage(self, capacity: int):
        array_shape = [capacity, *self.shape]
        # not all bit representations are valid for bool
        if self.dtype == bool:
            return torch.zeros(array_shape, dtype=torch.bool)
        return torch.from_numpy(np.empty(array_shape, dtype=self.dtype))

    def input_to_storage(self, input):
        return torch.from_numpy(np.array(input, dtype=self.dtype))

    def sample_to_output(self, sample):
        # sample has shape (batch_size, stack_size, obs_shape) right now, so
        # reshape to (batch_size, obs_shape, stack_size)
        perm = [0] + list(range(2, len(self.shape) + 2)) + [1]
        output = sample.permute(*perm)
        # squeeze the stack dim if it is 1
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


@dataclass
class IDListMetadata(ElementMetadata):
    """
    Internal representation is a np.array of Dict[str, np.array of type int64]
    Output is Dict[str, Tuple[np.array of type int32, np.array of type int64]], same as id_list in FeatureStore.
    The tuple is (offset, ids).
    TODO: implement for stack size > 1
    """

    keys: List[str]

    @classmethod
    def create_from_example(cls, name: str, example):
        res = cls(list(example.keys()))
        res.validate(name, example)
        return res

    def zero_example(self):
        return {k: [] for k in self.keys}

    def validate(self, name: str, input):
        assert isinstance(input, dict), f"{name}: {type(input)} isn't dict"
        for k, v in input.items():
            assert isinstance(k, str), f"{name}: {k} ({type(k)}) is not str"
            assert k in self.keys, f"{name}: {k} not in {self.keys}"
            arr = np.array(v)
            if len(arr) > 0:
                assert arr.dtype == np.int64, (
                    f"{name}: {v} arr has dtype {arr.dtype}, not np.int64"
                )

    def create_storage(self, capacity: int):
        array_shape = (capacity,)
        return np.empty(array_shape, dtype=object)

    def input_to_storage(self, input):
        return input

    def sample_to_output(self, sample):
        sample = sample.squeeze(1)
        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k in self.keys:
            offsets = []
            ids = []
            for elem in sample:
                # uninitialized case (when sampling next)
                if elem is None:
                    cur_ids = []
                else:
                    cur_ids = elem[k]
                offsets.append(len(ids))
                ids.extend(cur_ids)
            result[k] = (
                torch.tensor(offsets, dtype=torch.int32),
                torch.tensor(ids, dtype=torch.int64),
            )
        return result


@dataclass
class IDScoreListMetadata(ElementMetadata):
    """
    Internal representation is a np.array of Dict[str, np.array of type int64]
    Output is Dict[str, Tuple[np.array of type int32, np.array of type int64, np.array of type np.float32]], same as id_list in FeatureStore.
    The tuple is (offset, ids, scores).
    TODO: implement for stack size > 1
    """

    keys: List[str]

    @classmethod
    def create_from_example(cls, name: str, example):
        res = cls(list(example.keys()))
        res.validate(name, example)
        return res

    def zero_example(self):
        return {k: ([], []) for k in self.keys}

    def validate(self, name: str, input):
        assert isinstance(input, dict), f"{name}: {type(input)} isn't dict"
        for k, v in input.items():
            assert isinstance(k, str), f"{name}: {k} ({type(k)}) is not str"
            assert k in self.keys, f"{name}: {k} not in {self.keys}"
            assert isinstance(v, tuple) and len(v) == 2, (
                f"{name}: {v} ({type(v)}) is not len 2 tuple"
            )
            ids = np.array(v[0])
            scores = np.array(v[1])
            assert len(ids) == len(scores), f"{name}: {len(ids)} != {len(scores)}"
            if len(ids) > 0:
                assert ids.dtype == np.int64, (
                    f"{name}: ids dtype {ids.dtype} isn't np.int64"
                )
                assert scores.dtype in (
                    np.float32,
                    np.float64,
                ), f"{name}: scores dtype {scores.dtype} isn't np.float32/64"

    def create_storage(self, capacity: int):
        array_shape = (capacity,)
        return np.empty(array_shape, dtype=object)

    def input_to_storage(self, input):
        return input

    def sample_to_output(self, sample):
        sample = sample.squeeze(1)
        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k in self.keys:
            offsets = []
            ids = []
            scores = []
            for elem in sample:
                # uninitialized case (when sampling next)
                if elem is None:
                    cur_ids, cur_scores = [], []
                else:
                    cur_ids, cur_scores = elem[k]
                assert len(cur_ids) == len(cur_scores), (
                    f"{len(cur_ids)} != {len(cur_scores)}"
                )
                offsets.append(len(ids))
                ids.extend(cur_ids)
                scores.extend(cur_scores)
            result[k] = (
                torch.tensor(offsets, dtype=torch.int32),
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor(scores, dtype=torch.float32),
            )
        return result


class ReplayElement(NamedTuple):
    # Describing contents of each field of replay memory.
    name: str
    metadata: ElementMetadata


def make_replay_element(name, example):
    assert not isinstance(example, torch.Tensor), "Input shouldn't be tensor"
    metadata = None
    for metadata_cls in [DenseMetadata, IDListMetadata, IDScoreListMetadata]:
        try:
            metadata = metadata_cls.create_from_example(name, example)
            break
        except Exception as e:
            logger.info(
                f"Failed attempt to create {metadata_cls} from ({name}) {example}: {e}"
            )

    if metadata is None:
        raise ValueError(f"Unable to deduce type for {name}: {example}")

    return ReplayElement(name, metadata)


# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = "$store$_"

# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4

REQUIRED_KEYS = ["observation", "action", "reward", "terminal"]


class ReplayBuffer:
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
        stack_size: int = 1,
        replay_capacity: int = 10000,
        batch_size: int = 1,
        return_everything_as_stack: bool = False,
        return_as_timeline_format: bool = False,
        update_horizon: int = 1,
        gamma: float = 0.99,
    ) -> None:
        """Initializes ReplayBuffer.
        Args:
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          return_everything_as_stack: bool, set True if we want everything,
             not just states, to be stacked too
          return_as_timeline_format: bool, when set True, next_(states, actions, etc.)
            is returned list format, like the output of TimelineOperator
          update_horizon: int, length of update ('n' in n-step update).
          gamma: int, the discount factor.
        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """
        if replay_capacity < update_horizon + stack_size:
            raise ValueError(
                "There is not enough capacity to cover update_horizon and stack_size."
            )

        if return_as_timeline_format:
            if update_horizon <= 1:
                logger.warn(
                    f"Pointless to set return_as_timeline_format when "
                    f"update_horizon ({update_horizon}) isn't > 1."
                    "But we'll support it anyways..."
                )

        self._initialized_buffer = False
        self._stack_size = stack_size
        self._return_everything_as_stack = return_everything_as_stack
        self._return_as_timeline_format = return_as_timeline_format
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma

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

        # to be initialized on first add (put here to please pyre)
        self._store: Dict[str, torch.Tensor] = {}
        self._storage_types: List[ReplayElement] = []
        self._batch_type = collections.namedtuple("filler", [])
        # have these for ease
        self._extra_keys: List[str] = []
        self._key_to_replay_elem: Dict[str, ReplayElement] = {}
        self._zero_transition = {}
        self._transition_elements = {}

    def initialize_buffer(self, **kwargs):
        """Initialize replay buffer based on first input"""
        kwarg_keys = set(kwargs.keys())
        assert set(REQUIRED_KEYS).issubset(kwarg_keys), (
            f"{kwarg_keys} doesn't contain all of {REQUIRED_KEYS}"
        )

        # arbitrary order for extra keys
        self._extra_keys = list(kwarg_keys - set(REQUIRED_KEYS))

        self._storage_types: List[ReplayElement] = [
            make_replay_element(k, kwargs[k]) for k in REQUIRED_KEYS + self._extra_keys
        ]
        self._key_to_replay_elem = {
            elem.name: elem for elem in self.get_storage_signature()
        }
        self._create_storage()
        self._transition_elements = self.get_transition_elements()
        self._batch_type = collections.namedtuple(
            "batch_type", self._transition_elements
        )
        self._zero_transition = {
            elem.name: elem.metadata.zero_example() for elem in self._storage_types
        }
        self._initialized_buffer = True

        logger.info(f"Initializing {self.__class__.__name__}...")
        logger.info(f"\t stack_size: {self._stack_size}")
        logger.info(f"\t replay_capacity: {self._replay_capacity}")
        logger.info(f"\t update_horizon: {self._update_horizon}")
        logger.info(f"\t gamma: {self._gamma}")
        logger.info("\t storage_types: ")
        for elem in self._storage_types:
            logger.info(f"\t\t {elem}")

    @property
    def size(self) -> int:
        return self._num_valid_indices

    def set_index_valid_status(self, idx: int, is_valid: bool):
        old_valid = self._is_index_valid[idx]
        if not old_valid and is_valid:
            self._num_valid_indices += 1
        elif old_valid and not is_valid:
            self._num_valid_indices -= 1
        assert self._num_valid_indices >= 0, f"{self._num_valid_indices} is negative"

        self._is_index_valid[idx] = is_valid

    def _create_storage(self) -> None:
        """Creates the numpy arrays used to store transitions."""
        for storage_element in self.get_storage_signature():
            self._store[storage_element.name] = storage_element.metadata.create_storage(
                self._replay_capacity
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
        return self._storage_types

    def _add_zero_transition(self) -> None:
        """Adds a padding transition filled with zeros (Used in episode beginnings)."""
        self._add(**self._zero_transition)

    def add(self, **kwargs):
        """Adds a transition to the replay memory.
        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.
        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.
        If the replay memory is at capacity the oldest transition will be discarded.

        Only accept kwargs, which must contain observation, action, reward, terminal
        as keys.
        """
        if not self._initialized_buffer:
            self.initialize_buffer(**kwargs)

        self._check_add_types(**kwargs)
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
        self._add(**kwargs)
        self._num_transitions_in_current_episode += 1

        # mark the next stack_size-1 as invalid (note cursor has advanced by 1)
        for i in range(self._stack_size - 1):
            idx = (self.cursor() + i) % self._replay_capacity
            self.set_index_valid_status(idx=idx, is_valid=False)

        if kwargs["terminal"]:
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

    def _add(self, **kwargs):
        """Internal add method to add to the storage arrays.
        Args:
          *args: All the elements in a transition.
        """
        self._check_args_length(**kwargs)
        elements = self.get_add_args_signature()
        for element in elements:
            kwargs[element.name] = element.metadata.input_to_storage(
                kwargs[element.name]
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

    def _check_args_length(self, **kwargs):
        """Check if args passed to the add method have the same length as storage.
        Args:
          *args: Args for elements used in storage.
        Raises:
          ValueError: If args have wrong length.
        """
        if len(kwargs) != len(self.get_add_args_signature()):
            raise ValueError(
                f"Add expects: {self.get_add_args_signature()}; received {kwargs}"
            )

    def _check_add_types(self, **kwargs):
        """Checks if args passed to the add method match those of the storage.
        Args:
          *args: Args whose types need to be validated.
        Raises:
          ValueError: If args have wrong shape or dtype.
        """
        self._check_args_length(**kwargs)

        for store_element in self.get_add_args_signature():
            arg_element = kwargs[store_element.name]
            store_element.metadata.validate(store_element.name, arg_element)

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
        assert valid_indices.ndim == 1, (
            f"Expecting 1D tensor since is_index_valid is 1D. Got {valid_indices}."
        )
        return self.sample_transition_batch(
            batch_size=len(valid_indices), indices=valid_indices
        )

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).
        If get_transition_elements has been overridden and defines elements not
        stored in self._store, None will be returned and it will be
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
            assert isinstance(indices, torch.Tensor), (
                f"Indices {indices} have type {type(indices)} instead of torch.Tensor"
            )
            indices = indices.type(dtype=torch.int64)
        assert len(indices) == batch_size

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
        for element_name in self._transition_elements:
            if element_name == "state":
                batch = self._get_batch_for_indices("observation", indices)
            elif element_name == "next_state":
                batch = self._get_batch_for_indices(
                    "observation", next_indices, steps_for_timeline_format
                )
            elif element_name == "indices":
                batch = indices
            elif element_name == "terminal":
                terminal_indices = (indices + steps - 1) % self._replay_capacity
                batch = self._store["terminal"][terminal_indices].to(torch.bool)
            elif element_name == "reward":
                if self._return_as_timeline_format or self._return_everything_as_stack:
                    batch = self._get_batch_for_indices(
                        "reward", indices, steps_for_timeline_format
                    )
                else:
                    batch = self._reduce_multi_step_reward(multistep_indices, steps)
            elif element_name == "step":
                batch = steps
            elif element_name in self._store:
                batch = self._get_batch_for_indices(element_name, indices)
            elif element_name.startswith("next_"):
                store_name = element_name[len("next_") :]
                assert store_name in self._store, (
                    f"{store_name} is not in {self._store.keys()}"
                )
                batch = self._get_batch_for_indices(
                    store_name, next_indices, steps_for_timeline_format
                )
            else:
                # We assume the other elements are filled in by the subclass.
                batch = None

            # always enables the batch_size dim
            if isinstance(batch, torch.Tensor) and batch.ndim == 1:
                batch = batch.unsqueeze(1)
            batch_arrays.append(batch)
        return self._batch_type(*batch_arrays)

    def _get_batch_for_indices(
        self, key: str, indices: torch.Tensor, steps: Optional[torch.Tensor] = None
    ):
        """Get batch for given key.
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
        """Get stack of transition data."""
        assert len(indices.shape) == 1, f"{indices.shape} not 1-dimensional"
        # calculate 2d array of indices of shape (batch_size, stack_size)
        # ith row contain indices in the stack of obs at indices[i]
        stack_indices = indices.unsqueeze(1) + torch.arange(-self._stack_size + 1, 1)
        stack_indices %= self._replay_capacity
        retval = self._store[key][stack_indices]
        return self._key_to_replay_elem[key].metadata.sample_to_output(retval)

    def _get_steps(self, multistep_indices: torch.Tensor) -> torch.Tensor:
        """Calculate trajectory length, defined to be the number of states
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

    def get_transition_elements(self):
        """Returns element names for sample_transition_batch."""
        extra_names = []
        for name in self._extra_keys:
            for prefix in ["", "next_"]:
                extra_names.append(f"{prefix}{name}")
        return [
            "state",
            "action",
            "reward",
            "next_state",
            "next_action",
            "next_reward",
            "terminal",
            "indices",
            "step",
            *extra_names,
        ]

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
