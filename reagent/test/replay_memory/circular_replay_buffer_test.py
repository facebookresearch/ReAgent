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
"""Tests for circular_replay_buffer.py."""


import tempfile
import unittest

import numpy as np
import numpy.testing as npt
import torch
from reagent.replay_memory import circular_replay_buffer


# Default parameters used when creating the replay memory.
OBSERVATION_SHAPE = (84, 84)
OBS_DTYPE = np.uint8
STACK_SIZE = 4
BATCH_SIZE = 32


class CheckpointableClass(object):
    def __init__(self):
        self.attribute = 0


class ReplayBufferTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._test_subdir = self.tmp_dir.name
        num_dims = 10
        self._test_observation = np.ones(num_dims) * 1
        self._test_action = np.ones(num_dims) * 2
        self._test_reward = np.ones(num_dims) * 3
        self._test_terminal = np.ones(num_dims) * 4
        self._test_add_count = np.array(7)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def testConstructor(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        self.assertEqual(memory.add_count, 0)

    def testAdd(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        self.assertEqual(memory.cursor(), 0)
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, action=0, reward=0, terminal=0)
        # Check if the cursor moved STACK_SIZE -1 padding adds + 1, (the one above).
        self.assertEqual(memory.cursor(), STACK_SIZE)

    def testExtraAdd(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        self.assertEqual(memory.cursor(), 0)
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(
            observation=zeros, action=0, reward=0, terminal=0, extra1=0, extra2=[0, 0]
        )

        with self.assertRaisesRegex(ValueError, "Add expects"):
            memory.add(observation=zeros, action=0, reward=0, terminal=0)
        # Check if the cursor moved STACK_SIZE -1 zeros adds + 1, (the one above).
        self.assertEqual(memory.cursor(), STACK_SIZE)

    def testLowCapacity(self):
        with self.assertRaisesRegex(ValueError, "There is not enough capacity"):
            circular_replay_buffer.ReplayBuffer(
                stack_size=10,
                replay_capacity=10,
                batch_size=BATCH_SIZE,
                update_horizon=1,
                gamma=1.0,
            )

        with self.assertRaisesRegex(ValueError, "There is not enough capacity"):
            circular_replay_buffer.ReplayBuffer(
                stack_size=5,
                replay_capacity=10,
                batch_size=BATCH_SIZE,
                update_horizon=10,
                gamma=1.0,
            )

        # We should be able to create a buffer that contains just enough for a
        # transition.
        circular_replay_buffer.ReplayBuffer(
            stack_size=5,
            replay_capacity=10,
            batch_size=BATCH_SIZE,
            update_horizon=5,
            gamma=1.0,
        )

    def testNSteprewardum(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE,
            replay_capacity=10,
            batch_size=BATCH_SIZE,
            update_horizon=5,
            gamma=1.0,
        )

        for i in range(50):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=0,
                reward=2.0,
                terminal=0,
            )

        for _i in range(100):
            batch = memory.sample_transition_batch()
            # Make sure the total reward is reward per step x update_horizon.
            self.assertEqual(batch[2][0], 10.0)

    def testSampleTransitionBatch(self):
        replay_capacity = 10
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1, replay_capacity=replay_capacity, batch_size=2
        )
        num_adds = 50  # The number of transitions to add to the memory.
        for i in range(num_adds):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, OBS_DTYPE),
                action=0,
                reward=0,
                terminal=i % 4,
            )  # Every 4 transitions is terminal.
        # Test sampling with default batch size.
        for _i in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)
        # Test changing batch sizes.
        for _i in range(1000):
            batch = memory.sample_transition_batch(BATCH_SIZE)
            self.assertEqual(batch[0].shape[0], BATCH_SIZE)
        # Verify we revert to default batch size.
        for _i in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)

        # Verify we can specify what indices to sample.
        indices = [1, 2, 3, 5, 8]
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE) for i in indices]
        )
        expected_next_states = (expected_states + 1) % replay_capacity
        # Because the replay buffer is circular, we can exactly compute what the
        # states will be at the specified indices by doing a little mod math:
        expected_states += num_adds - replay_capacity
        expected_next_states += num_adds - replay_capacity
        # This is replicating the formula that was used above to determine what
        # transitions are terminal when adding observation (i % 4).
        expected_terminal = np.expand_dims(
            np.array([min((x + num_adds - replay_capacity) % 4, 1) for x in indices]), 1
        ).astype(bool)
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(batch.action, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_action, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_state, expected_next_states)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))

    def testSampleTransitionBatchExtra(self):
        replay_capacity = 10
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1, replay_capacity=replay_capacity, batch_size=2
        )
        num_adds = 50  # The number of transitions to add to the memory.
        for i in range(num_adds):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=0,
                reward=0,
                terminal=i % 4,
                extra1=i % 2,
                extra2=[i % 2, 0],
            )  # Every 4 transitions is terminal.
        # Test sampling with default batch size.
        for _i in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)
        # Test changing batch sizes.
        for _i in range(1000):
            batch = memory.sample_transition_batch(BATCH_SIZE)
            self.assertEqual(batch[0].shape[0], BATCH_SIZE)
        # Verify we revert to default batch size.
        for _i in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)

        # Verify we can specify what indices to sample.
        indices = [1, 2, 3, 5, 8]
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE) for i in indices]
        )
        expected_next_states = (expected_states + 1) % replay_capacity
        # Because the replay buffer is circular, we can exactly compute what the
        # states will be at the specified indices by doing a little mod math:
        expected_states += num_adds - replay_capacity
        expected_next_states += num_adds - replay_capacity
        # This is replicating the formula that was used above to determine what
        # transitions are terminal when adding observation (i % 4).
        expected_terminal = np.expand_dims(
            np.array([min((x + num_adds - replay_capacity) % 4, 1) for x in indices]), 1
        ).astype(bool)
        expected_extra1 = np.expand_dims(
            np.array([(x + num_adds - replay_capacity) % 2 for x in indices]), 1
        )
        expected_next_extra1 = np.expand_dims(
            np.array([(x + 1 + num_adds - replay_capacity) % 2 for x in indices]), 1
        )
        expected_extra2 = np.stack(
            [
                [(x + num_adds - replay_capacity) % 2 for x in indices],
                np.zeros((len(indices),)),
            ],
            axis=1,
        )
        expected_next_extra2 = np.stack(
            [
                [(x + 1 + num_adds - replay_capacity) % 2 for x in indices],
                np.zeros((len(indices),)),
            ],
            axis=1,
        )
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(batch.action, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_action, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_state, expected_next_states)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))
        npt.assert_array_equal(batch.extra1, expected_extra1)
        npt.assert_array_equal(batch.next_extra1, expected_next_extra1)
        npt.assert_array_equal(batch.extra2, expected_extra2)
        npt.assert_array_equal(batch.next_extra2, expected_next_extra2)

    def testSamplingWithterminalInTrajectory(self):
        replay_capacity = 10
        update_horizon = 3
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1,
            replay_capacity=replay_capacity,
            batch_size=2,
            update_horizon=update_horizon,
            gamma=1.0,
        )
        for i in range(replay_capacity):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=i * 2,
                reward=i,
                terminal=1 if i == 3 else 0,
            )
        indices = [2, 3, 4]
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        # In commone shape, state is 2-D unless stack_size > 1.
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE) for i in indices]
        )
        # The reward in the replay buffer will be (an asterisk marks the terminal
        # state):
        #   [0 1 2 3* 4 5 6 7 8 9]
        # Since we're setting the update_horizon to 3, the accumulated trajectory
        # reward starting at each of the replay buffer positions will be:
        #   [3 6 5 3 15 18 21 24]
        # Since indices = [2, 3, 4], our expected reward are [5, 3, 15].
        expected_reward = np.array([[5], [3], [15]])
        # Because update_horizon = 3, both indices 2 and 3 include terminal.
        expected_terminal = np.array([[1], [1], [0]]).astype(bool)
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(
            batch.action, np.expand_dims(np.array(indices) * 2, axis=1)
        )
        npt.assert_array_equal(batch.reward, expected_reward)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))

    def testIsTransitionValid(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=10, batch_size=2
        )

        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE),
            action=0,
            reward=0,
            terminal=0,
        )
        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE),
            action=0,
            reward=0,
            terminal=0,
        )
        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE),
            action=0,
            reward=0,
            terminal=1,
        )

        # These valids account for the automatically applied padding (3 blanks each
        # episode.
        # correct_valids = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        # The above comment is for the original Dopamine buffer, which doesn't
        # account for terminal frames within the update_horizon frames before
        # the cursor. In this case, the frame right before the cursor
        # is terminal, so even though it is within [c-update_horizon, c],
        # it should still be valid for sampling, as next state doesn't matter.
        correct_valids = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # The cursor is:                    ^\
        for i in range(10):
            self.assertEqual(
                correct_valids[i],
                memory.is_valid_transition(i),
                "Index %i should be %s" % (i, bool(correct_valids[i])),
            )


"""
Since we don't use saving, not maintaining for now
    def testSave(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        memory.observation = self._test_observation
        memory.action = self._test_action
        memory.reward = self._test_reward
        memory.terminal = self._test_terminal
        current_iteration = 5
        stale_iteration = current_iteration - circular_replay_buffer.CHECKPOINT_DURATION
        memory.save(self._test_subdir, stale_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            stale_filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, stale_iteration)
            )
            self.assertTrue(os.path.exists(stale_filename))

        memory.save(self._test_subdir, current_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, current_iteration)
            )
            self.assertTrue(os.path.exists(filename))
            # The stale version file should have been deleted.
            self.assertFalse(os.path.exists(stale_filename))

    def testSaveNonNDArrayAttributes(self):
        # Tests checkpointing an attribute which is not a numpy array.
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )

        # Add some non-numpy data: an int, a string, an object.
        memory.dummy_attribute_1 = 4753849
        memory.dummy_attribute_2 = "String data"
        memory.dummy_attribute_3 = CheckpointableClass()

        current_iteration = 5
        stale_iteration = current_iteration - circular_replay_buffer.CHECKPOINT_DURATION
        memory.save(self._test_subdir, stale_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            stale_filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, stale_iteration)
            )
            self.assertTrue(os.path.exists(stale_filename))

        memory.save(self._test_subdir, current_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, current_iteration)
            )
            self.assertTrue(os.path.exists(filename))
            # The stale version file should have been deleted.
            self.assertFalse(os.path.exists(stale_filename))

    def testLoadFromNonexistentDirectory(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        # We are trying to load from a non-existent directory, so a NotFoundError
        # will be raised.
        with self.assertRaises(FileNotFoundError):
            memory.load("/does/not/exist", "3")
        self.assertNotEqual(memory._store["observation"], self._test_observation)
        self.assertNotEqual(memory._store["action"], self._test_action)
        self.assertNotEqual(memory._store["reward"], self._test_reward)
        self.assertNotEqual(memory._store["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_count, self._test_add_count)

    def testPartialLoadFails(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        self.assertNotEqual(memory._store["observation"], self._test_observation)
        self.assertNotEqual(memory._store["action"], self._test_action)
        self.assertNotEqual(memory._store["reward"], self._test_reward)
        self.assertNotEqual(memory._store["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_count, self._test_add_count)
        numpy_arrays = {
            "observation": self._test_observation,
            "action": self._test_action,
            "terminal": self._test_terminal,
            "add_count": self._test_add_count,
        }
        for attr in numpy_arrays:
            filename = os.path.join(self._test_subdir, "{}_ckpt.3.gz".format(attr))
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    np.save(outfile, numpy_arrays[attr], allow_pickle=False)
        # We are are missing the reward file, so a NotFoundError will be raised.
        with self.assertRaises(FileNotFoundError):
            memory.load(self._test_subdir, "3")
        # Since we are missing the reward file, it should not have loaded any of
        # the other files.
        self.assertNotEqual(memory._store["observation"], self._test_observation)
        self.assertNotEqual(memory._store["action"], self._test_action)
        self.assertNotEqual(memory._store["reward"], self._test_reward)
        self.assertNotEqual(memory._store["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_count, self._test_add_count)

    def testLoad(self):
        memory = circular_replay_buffer.ReplayBuffer(
            observation_shape=OBSERVATION_SHAPE,
            stack_size=STACK_SIZE,
            replay_capacity=5,
            batch_size=BATCH_SIZE,
        )
        self.assertNotEqual(memory._store["observation"], self._test_observation)
        self.assertNotEqual(memory._store["action"], self._test_action)
        self.assertNotEqual(memory._store["reward"], self._test_reward)
        self.assertNotEqual(memory._store["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_count, self._test_add_count)
        store_prefix = "$store$_"
        numpy_arrays = {
            store_prefix + "observation": self._test_observation,
            store_prefix + "action": self._test_action,
            store_prefix + "reward": self._test_reward,
            store_prefix + "terminal": self._test_terminal,
            "add_count": self._test_add_count,
        }
        for attr in numpy_arrays:
            filename = os.path.join(self._test_subdir, "{}_ckpt.3.gz".format(attr))
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    np.save(outfile, numpy_arrays[attr], allow_pickle=False)
        memory.load(self._test_subdir, "3")
        npt.assert_allclose(memory._store["observation"], self._test_observation)
        npt.assert_allclose(memory._store["action"], self._test_action)
        npt.assert_allclose(memory._store["reward"], self._test_reward)
        npt.assert_allclose(memory._store["terminal"], self._test_terminal)
        self.assertEqual(memory.add_count, self._test_add_count)
"""
