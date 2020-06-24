#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
import numpy.testing as npt
import torch
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer, ReplayElement
from reagent.test.base.horizon_test_base import HorizonTestBase


logger = logging.getLogger(__name__)
torch.set_printoptions(profile="full")

OBS_SHAPE = (3, 3)
OBS_TYPE = np.float32

"""
Everything about the MDP, except terminal states, are defined here.
Terminal states are derived from Trajectory lengths array.
"""


def get_add_transition(i):
    """ For adding into RB """
    return {
        "state": np.ones(OBS_SHAPE) * i,
        "action": int(i),
        "reward": float(2 * i),
        "extra1": float(3 * i),
    }


ZERO_FEATURES = {
    "state": np.zeros(OBS_SHAPE),
    "action": int(0),
    "reward": float(0),
    "extra1": float(0),
}


def get_stacked_transition(i, stack_size, traj_start_idx):
    """ For getting expected stacked state of i """
    res = {k: [] for k in ["state", "action", "reward", "extra1"]}
    # must pad with some zero states
    for idx in range(i - stack_size + 1, i + 1):
        trans = ZERO_FEATURES if idx < traj_start_idx else get_add_transition(idx)
        for k in res:
            res[k].append(trans[k])

    return {k: np.stack(v, axis=-1) for k, v in res.items()}


def setup_buffer(buffer_size, trajectory_lengths, stack_size=None, multi_steps=None):
    """ We will insert one trajectory into the RB. """
    stack_size = stack_size if stack_size is not None else 1
    update_horizon = multi_steps if multi_steps is not None else 1
    memory = ReplayBuffer(
        observation_shape=OBS_SHAPE,
        observation_dtype=OBS_TYPE,
        stack_size=stack_size,
        replay_capacity=buffer_size,
        batch_size=1,
        update_horizon=update_horizon,
        return_everything_as_stack=stack_size is not None,
        return_as_timeline_format=multi_steps is not None,
        extra_storage_types=[ReplayElement("extra1", (), np.float32)],
    )

    i = 0
    for traj_len in trajectory_lengths:
        for j in range(traj_len):
            trans = get_add_transition(i)
            terminal = bool(j == traj_len - 1)
            memory.add(
                trans["state"],
                trans["action"],
                trans["reward"],
                terminal,
                trans["extra1"],
            )
            i += 1
    return memory.sample_all_valid_transitions()


def generic_stack_test_helper(buffer_size, trajectory_lengths, stack_size):
    batch = setup_buffer(buffer_size, trajectory_lengths, stack_size=stack_size)

    expected = {k: [] for k in ["state", "action", "reward", "extra1"]}
    terminal_array = []

    actual_rb_index = stack_size - 1
    i = 0
    for traj_len in trajectory_lengths:
        traj_start = i
        for j in range(traj_len):
            cur = get_stacked_transition(i, stack_size, traj_start)
            for k in expected:
                expected[k].append(cur[k])
            terminal_array.append(bool(j == traj_len - 1))

            actual_rb_index += 1
            i += 1
        actual_rb_index += stack_size - 1

    expected["terminal"] = np.expand_dims(terminal_array, axis=1)
    for k in expected:
        expected[k] = torch.tensor(expected[k])

    for k in expected:
        batch_val = getattr(batch, k)
        npt.assert_array_equal(
            batch_val,
            expected[k],
            err_msg=f"key {k}; expected {expected[k]}, got {batch_val}",
            verbose=True,
        )


def generic_stack_multi_steps_test_helper(
    buffer_size, trajectory_lengths, stack_size, multi_steps
):
    batch = setup_buffer(
        buffer_size, trajectory_lengths, stack_size=stack_size, multi_steps=multi_steps
    )

    # start with state, action, extra1 (these are single)
    expected = {k: [] for k in ["state", "action", "extra1"]}
    terminal_array = []

    actual_rb_index = stack_size - 1
    i = 0
    for traj_len in trajectory_lengths:
        traj_start = i
        for j in range(traj_len):
            cur = get_stacked_transition(i, stack_size, traj_start)
            for k in expected:
                expected[k].append(cur[k])
            terminal_array.append(bool(j >= traj_len - multi_steps))

            actual_rb_index += 1
            i += 1
        actual_rb_index += stack_size - 1
    assert (
        actual_rb_index <= buffer_size
    ), f"{actual_rb_index} is larger than {buffer_size}"

    expected["terminal"] = np.expand_dims(terminal_array, axis=1)
    for k in expected:
        expected[k] = torch.tensor(expected[k])
    batch_size = expected["state"].shape[0]

    for k in expected:
        batch_val = getattr(batch, k)
        npt.assert_array_equal(
            batch_val,
            expected[k],
            err_msg=f"key {k}; expected {expected[k]}, got {batch_val}",
            verbose=True,
        )

    # now examine reward, next_state, next_action, next_extra1, which would be lists of size step
    expected = {k: [] for k in ["reward", "next_state", "next_action", "next_extra1"]}
    i = 0
    for traj_len in trajectory_lengths:
        traj_start = i
        for _ in range(traj_len):
            multistep_trans = {k: [] for k in expected}

            # rewards start at current
            traj_end = traj_start + traj_len
            for j in range(i, i + multi_steps):
                if j < traj_end:
                    stacked_trans = get_stacked_transition(j, stack_size, traj_start)
                    multistep_trans["reward"].append(stacked_trans["reward"])

            # next features start at current + 1
            for j in range(i + 1, i + multi_steps + 1):
                if j <= traj_end:
                    stacked_trans = get_stacked_transition(j, stack_size, traj_start)
                    for k in ["next_state", "next_action", "next_extra1"]:
                        stripped_k = k[len("next_") :]
                        multistep_trans[k].append(stacked_trans[stripped_k])

            multistep_trans = {k: torch.tensor(v) for k, v in multistep_trans.items()}
            for k in expected:
                expected[k].append(multistep_trans[k])

            i += 1

    # validate the lengths
    for k in expected:
        batch_val = getattr(batch, k)
        assert isinstance(batch_val, list), f"batch[{k}] has type {type(batch_val)}"
        assert len(batch_val) == batch_size
        for i in range(batch_size):
            assert isinstance(
                batch_val[i], torch.Tensor
            ), f"batch[{k}][{i}] has type {type(batch_val[i])};\n{batch_val}"
            assert batch_val[i].shape[0] == batch.step[i], (
                f"batch[{k}][{i}] {batch_val[i].shape} doesn't start "
                f"with {batch.step[i]};\n{batch_val}"
            )
            # sanity check
            assert len(expected[k][i]) == batch.step[i], (
                f"expected[{k}][{i}] {expected[k][i]} with len {len(expected[k][i])} should have len "
                f"{batch.step[i]};\n{expected[k]}, {batch_val}"
            )

    for k in expected:
        batch_val = getattr(batch, k)
        for i in range(batch_size):
            # NOTE: the last transition for terminals is undefined
            A = batch_val[i]
            B = expected[k][i]
            if batch.terminal[i]:
                A = A[:-1]
                B = B[:-1]

            npt.assert_array_equal(A, B)


MAX_TRAJ_LEN = 100
NUM_TRAJ_LIMIT = 10


class ExtraReplayBufferTest(HorizonTestBase):
    """ Stress tests for the replay buffer, especially for new flags. """

    def test_stack_slaughter(self):
        stack_size = 7
        for i in range(1, NUM_TRAJ_LIMIT):
            traj_lengths = torch.randint(1, MAX_TRAJ_LEN, (i,))
            buffer_size = (traj_lengths.sum() + (i + 1) * (stack_size - 1)).item()
            logger.info(
                f"Inserting {i} trajectories...\nArguments are: "
                f"buffer_size:{buffer_size}, "
                f"traj_lengths:{traj_lengths}, "
                f"stack_size:{stack_size}"
            )
            generic_stack_test_helper(buffer_size, traj_lengths.tolist(), stack_size)
            logger.info(f"Inserting {i} trajectories passed...")

    def test_stack_multistep_flags_slaughter(self):
        stack_size = 5
        multi_steps = 6
        for i in range(1, NUM_TRAJ_LIMIT):
            traj_lengths = torch.randint(1, MAX_TRAJ_LEN, (i,))
            buffer_size = (traj_lengths.sum() + (i + 1) * (stack_size - 1)).item()
            # handle edge case which would raise ValueError
            if buffer_size < stack_size + multi_steps:
                buffer_size = stack_size + multi_steps
            logger.info(
                f"Inserting {i} trajectories...\nArguments are: "
                f"buffer_size:{buffer_size}, "
                f"traj_lengths:{traj_lengths}, "
                f"stack_size:{stack_size}, "
                f"multi_steps:{multi_steps}"
            )
            generic_stack_multi_steps_test_helper(
                buffer_size, traj_lengths.tolist(), stack_size, multi_steps
            )
            logger.info(f"Inserting {i} trajectories passed...")

    def test_replay_overflow(self):
        """
        hard to make a stress test for this, since tracking which indices
        gets replaced would be effectively building a second RB
        so instead opt for simple test...
        stack_size = 2 so there's 1 padding.
        """
        multi_steps = 2
        stack_size = 2
        memory = ReplayBuffer(
            observation_shape=OBS_SHAPE,
            observation_dtype=OBS_TYPE,
            stack_size=stack_size,
            replay_capacity=6,
            batch_size=1,
            update_horizon=multi_steps,
            return_everything_as_stack=None,
            return_as_timeline_format=True,
        )

        def trans(i):
            return np.ones(OBS_SHAPE, dtype=OBS_TYPE), int(2 * i), float(3 * i)

        # Contents of RB
        # start: [X, X, X, X, X, X]
        npt.assert_array_equal(
            memory._is_index_valid, [False, False, False, False, False, False]
        )

        # t0: [X, s0, X, X, X, X]
        memory.add(*trans(0), False)
        npt.assert_array_equal(
            memory._is_index_valid, [False, False, False, False, False, False]
        )

        # t1: [X, s0, s1, X, X, X]
        memory.add(*trans(1), False)
        npt.assert_array_equal(
            memory._is_index_valid, [False, False, False, False, False, False]
        )

        # t2: [X, s0, s1, s2, X, X]
        # s0 finally becomes valid as its next state was added
        memory.add(*trans(2), False)
        npt.assert_array_equal(
            memory._is_index_valid, [False, True, False, False, False, False]
        )
        batch = memory.sample_all_valid_transitions()
        npt.assert_array_equal(batch.action, [[0, 0]])
        npt.assert_array_equal(batch.next_action[0], [[0, 2], [2, 4]])

        # t3: [X, s0, s1, s2, s3, X]
        # episode termination validates whole episode
        memory.add(*trans(3), True)
        npt.assert_array_equal(
            memory._is_index_valid, [False, True, True, True, True, False]
        )
        batch = memory.sample_all_valid_transitions()
        npt.assert_array_equal(batch.action, [[0, 0], [0, 2], [2, 4], [4, 6]])
        npt.assert_array_equal(batch.next_action[0], [[0, 2], [2, 4]])
        npt.assert_array_equal(batch.next_action[1], [[2, 4], [4, 6]])
        # batch.next_action[2][1] is garbage
        npt.assert_array_equal(batch.next_action[2][0], [4, 6])
        # batch.next_action[3] is [garbage]

        # t4: [s4, s0, s1, s2, s3, X]
        # s0 invalidated as its previous frame is corrupted
        memory.add(*trans(4), False)
        npt.assert_array_equal(
            memory._is_index_valid, [False, False, True, True, True, False]
        )
        batch = memory.sample_all_valid_transitions()
        npt.assert_array_equal(batch.action, [[0, 2], [2, 4], [4, 6]])
        npt.assert_array_equal(batch.next_action[0], [[2, 4], [4, 6]])
        npt.assert_array_equal(batch.next_action[1][0], [4, 6])

        # t5: [s4, s5, s1, s2, s3, X]
        memory.add(*trans(5), False)
        npt.assert_array_equal(
            memory._is_index_valid, [False, False, False, True, True, False]
        )
        batch = memory.sample_all_valid_transitions()
        npt.assert_array_equal(batch.action, [[2, 4], [4, 6]])
        npt.assert_array_equal(batch.next_action[0][0], [4, 6])

        # t6: [s4, s5, s6, s2, s3, X]
        memory.add(*trans(6), True)
        npt.assert_array_equal(
            memory._is_index_valid, [True, True, True, False, True, False]
        )
        batch = memory.sample_all_valid_transitions()
        npt.assert_array_equal(batch.action, [[0, 8], [8, 10], [10, 12], [4, 6]])
        npt.assert_array_equal(batch.next_action[0], [[8, 10], [10, 12]])
        npt.assert_array_equal(batch.next_action[1][0], [10, 12])
        # batch.next_action[2] is [garbage]
        # batch.next_action[3] is [garbage]

        logger.info("Overflow test passes!")
