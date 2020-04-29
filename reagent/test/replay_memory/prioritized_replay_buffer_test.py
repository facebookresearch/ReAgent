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
"""Tests for prioritzed replay memory."""

import unittest

import numpy as np
from reagent.replay_memory import prioritized_replay_buffer


# Default parameters used when creating the replay memory.
SCREEN_SIZE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32
REPLAY_CAPACITY = 100


class PrioritizedReplayBufferTest(unittest.TestCase):
    def create_default_memory(self):
        return prioritized_replay_buffer.PrioritizedReplayBuffer(
            SCREEN_SIZE, STACK_SIZE, REPLAY_CAPACITY, BATCH_SIZE, max_sample_attempts=10
        )  # For faster tests.

    def add_blank(self, memory, action=0, reward=0.0, terminal=0, priority=1.0):
        """Adds a replay transition with a blank observation.
        Allows setting action, reward, terminal.
        Args:
          memory: The replay memory.
          action: Integer.
          reward: Float.
          terminal: Integer (0 or 1).
          priority: Float. Defults to standard priority of 1.
        Returns:
          Index of the transition just added.
        """
        dummy = np.zeros(SCREEN_SIZE)
        memory.add(dummy, action, reward, terminal, priority)
        index = (memory.cursor() - 1) % REPLAY_CAPACITY
        return index

    def testAddWithAndWithoutPriority(self):
        memory = self.create_default_memory()
        self.assertEqual(memory.cursor(), 0)
        zeros = np.zeros(SCREEN_SIZE)

        self.add_blank(memory)
        self.assertEqual(memory.cursor(), STACK_SIZE)
        self.assertEqual(memory.add_count, STACK_SIZE)

        # Check that the prioritized replay buffer expects an additional argument
        # for priority.
        with self.assertRaisesRegex(ValueError, "Add expects"):
            memory.add(zeros, 0, 0, 0)

    def testDummyScreensAddedToNewMemory(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        for i in range(index):
            self.assertEqual(memory.sum_tree.get(i), 0.0)

    def testGetPriorityWithInvalidIndices(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        with self.assertRaises(AssertionError, msg="Indices must be an array."):
            memory.get_priority(index)
        with self.assertRaises(
            AssertionError, msg="Indices must be int32s, given: int64"
        ):
            memory.get_priority(np.array([index]))

    def testSetAndGetPriority(self):
        memory = self.create_default_memory()
        batch_size = 7
        indices = np.zeros(batch_size, dtype=np.int32)
        for index in range(batch_size):
            indices[index] = self.add_blank(memory)
        priorities = np.arange(batch_size)
        memory.set_priority(indices, priorities)
        # We send the indices in reverse order and verify the priorities come back
        # in that same order.
        fetched_priorities = memory.get_priority(np.flip(indices, 0))
        for i in range(batch_size):
            self.assertEqual(priorities[i], fetched_priorities[batch_size - 1 - i])

    def testNewElementHasHighPriority(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        self.assertEqual(memory.get_priority(np.array([index], dtype=np.int32))[0], 1.0)

    def testLowPriorityElementNotFrequentlySampled(self):
        memory = self.create_default_memory()
        # Add an item and set its priority to 0.
        self.add_blank(memory, terminal=0, priority=0.0)
        # Now add a few new items.
        for _ in range(3):
            self.add_blank(memory, terminal=1)
        # This test should always pass.
        for _ in range(100):
            _, _, _, _, _, _, terminals, _, _ = memory.sample_transition_batch(
                batch_size=2
            )
            # Ensure all terminals are set to 1.
            self.assertTrue((terminals == 1).all())

    def testSampleIndexBatchTooManyFailedRetries(self):
        memory = self.create_default_memory()
        # Only adding a single observation is not enough to be able to sample
        # (as it both straddles the cursor and does not pass the
        # `index >= self.cursor() - self._update_horizon` check in
        # circular_replay_buffer.py).
        self.add_blank(memory)
        with self.assertRaises(
            RuntimeError,
            msg="Max sample attempts: Tried 10 times but only sampled 1 valid "
            "indices. Batch size is 2",
        ):
            memory.sample_index_batch(2)

    def testSampleIndexBatch(self):
        memory = prioritized_replay_buffer.PrioritizedReplayBuffer(
            SCREEN_SIZE,
            STACK_SIZE,
            REPLAY_CAPACITY,
            BATCH_SIZE,
            max_sample_attempts=REPLAY_CAPACITY,
        )
        # This will ensure we end up with cursor == 1.
        for _ in range(REPLAY_CAPACITY - STACK_SIZE + 2):
            self.add_blank(memory)
        self.assertEqual(memory.cursor(), 1)
        samples = memory.sample_index_batch(REPLAY_CAPACITY)
        # Because cursor == 1, the invalid range as set by circular_replay_buffer.py
        # will be # [0, 1, 2, 3], resulting in all samples being in
        # [STACK_SIZE, REPLAY_CAPACITY - 1].
        for sample in samples:
            self.assertGreaterEqual(sample, STACK_SIZE)
            self.assertLessEqual(sample, REPLAY_CAPACITY - 1)
