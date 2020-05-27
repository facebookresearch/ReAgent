#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import gym
import numpy as np
import numpy.testing as npt
from reagent.gym.preprocessors import make_replay_buffer_inserter
from reagent.gym.types import Transition
from reagent.replay_memory import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase


logger = logging.getLogger(__name__)

try:
    from recsim.environments import interest_evolution, interest_exploration

    HAS_RECSIM = True
except ModuleNotFoundError:
    HAS_RECSIM = False


def _create_replay_buffer_and_insert(env: gym.Env):
    env.seed(1)
    replay_buffer = ReplayBuffer.create_from_env(
        env, replay_memory_size=6, batch_size=1
    )
    replay_buffer_inserter = make_replay_buffer_inserter(env)
    obs = env.reset()
    inserted = []
    terminal = False
    i = 0
    while not terminal and i < 5:
        logger.info(f"Iteration: {i}")
        action = env.action_space.sample()
        next_obs, reward, terminal, _ = env.step(action)
        inserted.append(
            {
                "observation": obs,
                "action": action,
                "reward": reward,
                "terminal": terminal,
            }
        )
        transition = Transition(
            mdp_id=0,
            sequence_number=i,
            observation=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            log_prob=0.0,
        )
        replay_buffer_inserter(replay_buffer, transition)
        obs = next_obs
        i += 1

    return replay_buffer, inserted


class TestBasicReplayBufferInserter(HorizonTestBase):
    def test_cartpole(self):
        env = gym.make("CartPole-v0")
        replay_buffer, inserted = _create_replay_buffer_and_insert(env)
        batch = replay_buffer.sample_transition_batch_tensor(indices=np.array([0]))
        npt.assert_array_almost_equal(
            inserted[0]["observation"], batch.state.squeeze(0)
        )
        npt.assert_array_almost_equal(
            inserted[1]["observation"], batch.next_state.squeeze(0)
        )
        npt.assert_array_equal(inserted[0]["action"], batch.action.squeeze(0))
        npt.assert_array_equal(inserted[1]["action"], batch.next_action.squeeze(0))


class TestRecSimReplayBufferInserter(HorizonTestBase):
    @unittest.skipIf(not HAS_RECSIM, "RecSim not installed")
    def test_recsim_interest_evolution(self):
        num_candidate = 10
        env_config = {
            "num_candidates": num_candidate,
            "slate_size": 3,
            "resample_documents": False,
            "seed": 1,
        }
        env = interest_evolution.create_environment(env_config)
        replay_buffer, inserted = _create_replay_buffer_and_insert(env)
        batch = replay_buffer.sample_transition_batch_tensor(indices=np.array([0]))
        npt.assert_array_almost_equal(
            inserted[0]["observation"]["user"], batch.state.squeeze(0)
        )
        npt.assert_array_almost_equal(
            inserted[1]["observation"]["user"], batch.next_state.squeeze(0)
        )
        docs = list(inserted[0]["observation"]["doc"].values())
        next_docs = list(inserted[1]["observation"]["doc"].values())
        for i in range(num_candidate):
            npt.assert_array_equal(docs[i], batch.doc.squeeze(0)[i])
            npt.assert_array_equal(next_docs[i], batch.next_doc.squeeze(0)[i])
        npt.assert_array_equal(inserted[0]["action"], batch.action.squeeze(0))
        npt.assert_array_equal(inserted[1]["action"], batch.next_action.squeeze(0))
        npt.assert_array_equal([0, 0, 0], batch.response_click.squeeze(0))
        npt.assert_array_equal([0, 0, 0], batch.response_cluster_id.squeeze(0))
        npt.assert_array_equal([0, 0, 0], batch.response_liked.squeeze(0))
        npt.assert_array_equal([0.0, 0.0, 0.0], batch.response_quality.squeeze(0))
        npt.assert_array_equal([0.0, 0.0, 0.0], batch.response_watch_time.squeeze(0))
        resp = inserted[1]["observation"]["response"]
        for i in range(env_config["slate_size"]):
            npt.assert_array_equal(
                resp[i]["click"], batch.next_response_click.squeeze(0)[i]
            )
            npt.assert_array_equal(
                resp[i]["cluster_id"], batch.next_response_cluster_id.squeeze(0)[i]
            )
            npt.assert_array_equal(
                resp[i]["liked"], batch.next_response_liked.squeeze(0)[i]
            )
            npt.assert_array_almost_equal(
                resp[i]["quality"], batch.next_response_quality.squeeze(0)[i]
            )
            npt.assert_array_almost_equal(
                resp[i]["watch_time"], batch.next_response_watch_time.squeeze(0)[i]
            )

    @unittest.skipIf(not HAS_RECSIM, "RecSim not installed")
    def test_recsim_interest_exploration(self):
        num_candidate = 10
        env_config = {
            "num_candidates": num_candidate,
            "slate_size": 3,
            "resample_documents": False,
            "seed": 1,
        }
        env = interest_exploration.create_environment(env_config)
        replay_buffer, inserted = _create_replay_buffer_and_insert(env)
        batch = replay_buffer.sample_transition_batch_tensor(indices=np.array([0]))
        npt.assert_array_almost_equal(
            inserted[0]["observation"]["user"].astype(np.float32),
            batch.state.squeeze(0),
        )
        npt.assert_array_almost_equal(
            inserted[1]["observation"]["user"], batch.next_state.squeeze(0)
        )
        docs = list(inserted[0]["observation"]["doc"].values())
        next_docs = list(inserted[1]["observation"]["doc"].values())
        for i in range(num_candidate):
            npt.assert_array_almost_equal(
                docs[i]["quality"], batch.doc_quality.squeeze(0)[i]
            )
            npt.assert_array_almost_equal(
                next_docs[i]["quality"], batch.next_doc_quality.squeeze(0)[i]
            )
        npt.assert_array_equal(inserted[0]["action"], batch.action.squeeze(0))
        npt.assert_array_equal(inserted[1]["action"], batch.next_action.squeeze(0))
        npt.assert_array_equal([0, 0, 0], batch.response_click.squeeze(0))
        npt.assert_array_equal([0, 0, 0], batch.response_cluster_id.squeeze(0))
        npt.assert_array_equal([0.0, 0.0, 0.0], batch.response_quality.squeeze(0))
        resp = inserted[1]["observation"]["response"]
        for i in range(env_config["slate_size"]):
            npt.assert_array_equal(
                resp[i]["click"], batch.next_response_click.squeeze(0)[i]
            )
            npt.assert_array_equal(
                resp[i]["cluster_id"], batch.next_response_cluster_id.squeeze(0)[i]
            )
            npt.assert_array_almost_equal(
                resp[i]["quality"].astype(np.float32),
                batch.next_response_quality.squeeze(0)[i],
            )
