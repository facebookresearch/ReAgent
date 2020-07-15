#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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
"""An implementation of Prioritized Experience Replay (PER).
This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo
Hessel for providing useful pointers on the algorithm and its implementation.
"""

import numpy as np
import torch
from reagent.replay_memory import circular_replay_buffer, sum_tree
from reagent.replay_memory.circular_replay_buffer import ReplayElement


class PrioritizedReplayBuffer(circular_replay_buffer.ReplayBuffer):
    """An out-of-graph Replay Buffer for Prioritized Experience Replay.
    See circular_replay_buffer.py for details.
    """

    def __init__(
        self,
        stack_size,
        replay_capacity,
        batch_size,
        update_horizon=1,
        gamma=0.99,
        max_sample_attempts=1000,
    ):
        """Initializes PrioritizedReplayBuffer.
        Args:
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          update_horizon: int, length of update ('n' in n-step update).
          gamma: int, the discount factor.
        """
        super(PrioritizedReplayBuffer, self).__init__(
            stack_size=stack_size,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma,
        )
        self._max_sample_attempts = max_sample_attempts
        self.sum_tree = sum_tree.SumTree(replay_capacity)

    def _add(self, **kwargs):
        """Internal add method to add to the underlying memory arrays.
        The arguments need to match add_arg_signature.
        If priority is none, it is set to the maximum priority ever seen.
        Args:
        """
        self._check_args_length(**kwargs)

        # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
        # to the maximum priority so far.
        # Picks out 'priority' from arguments and adds it to the sum_tree.
        transition = {}
        for element in self.get_add_args_signature():
            if element.name == "priority":
                priority = kwargs[element.name]
            else:
                transition[element.name] = element.metadata.input_to_storage(
                    kwargs[element.name]
                )

        self.sum_tree.set(self.cursor(), priority)
        super(PrioritizedReplayBuffer, self)._add_transition(transition)

    def sample_index_batch(self, batch_size: int) -> torch.Tensor:
        """Returns a batch of valid indices sampled as in Schaul et al. (2015).
        Args:
          batch_size: int, number of indices returned.
        Returns:
          1D tensor of ints, a batch of valid indices sampled uniformly.
        Raises:
          Exception: If the batch was not constructed after maximum number of tries.
        """
        # TODO: do priority sampling with torch as well.
        # Sample stratified indices. Some of them might be invalid.
        indices = self.sum_tree.stratified_sample(batch_size)
        allowed_attempts = self._max_sample_attempts
        for i in range(len(indices)):
            if not self.is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        "Max sample attempts: Tried {} times but only sampled {}"
                        " valid indices. Batch size is {}".format(
                            self._max_sample_attempts, i, batch_size
                        )
                    )
                index = indices[i]
                while not self.is_valid_transition(index) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self.sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index
        return torch.tensor(indices, dtype=torch.int64)

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions with extra storage and the priorities.
        The extra storage are defined through the extra_storage_types constructor
        argument.
        When the transition is terminal next_state_batch has undefined contents.
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or 1D tensor of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        """
        transition = super(PrioritizedReplayBuffer, self).sample_transition_batch(
            batch_size, indices
        )
        # The parent returned an empty array for the probabilities. Fill it with the
        # contents of the sum tree. Note scalar values are returned as (batch_size, 1).

        batch_arrays = []
        for element_name in self._transition_elements:
            if element_name == "sampling_probabilities":
                batch = torch.from_numpy(
                    self.get_priority(transition.indices.numpy().astype(np.int32))
                ).view(batch_size, 1)
            else:
                batch = getattr(transition, element_name)
            batch_arrays.append(batch)

        return self._batch_type(*batch_arrays)

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.
        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
          priorities: float, the corresponding priorities.
        """
        assert (
            indices.dtype == np.int32
        ), "Indices must be integers, " "given: {}".format(indices.dtype)
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.
        For any memory location not yet used, the corresponding priority is 0.
        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
        Returns:
          priorities: float, the corresponding priorities.
        """
        assert indices.shape, "Indices must be an array."
        assert indices.dtype == np.int32, "Indices must be int32s, " "given: {}".format(
            indices.dtype
        )
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return priority_batch

    def get_transition_elements(self):
        parent_transition_elements = super(
            PrioritizedReplayBuffer, self
        ).get_transition_elements()
        return parent_transition_elements + ["sampling_probabilities"]
