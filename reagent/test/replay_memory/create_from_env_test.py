#!/usr/bin/env python3

# pyre-unsafe

import logging
import unittest

import numpy as np
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)

try:
    from reagent.gym.envs import RecSim

    HAS_RECSIM = True
except ImportError as e:
    logger.info(f"Exception {e}")
    HAS_RECSIM = False


class CreateFromEnvTest(unittest.TestCase):
    @unittest.skipIf(not HAS_RECSIM, "recsim is not installed")
    def test_create_from_recsim_interest_exploration(self):
        env = RecSim(
            num_candidates=20,
            slate_size=3,
            resample_documents=False,
            is_interest_exploration=True,
        )
        replay_buffer = ReplayBuffer(replay_capacity=100, batch_size=10)
        obs = env.reset()
        observation = obs["user"]
        action = env.action_space.sample()
        log_prob = -1.0
        quality = np.stack([v["quality"] for v in obs["doc"].values()], axis=0)
        cluster_id = np.array([v["cluster_id"] for v in obs["doc"].values()])

        next_obs, reward, terminal, _env = env.step(action)

        response = next_obs["response"]
        click = np.array([r["click"] for r in response])
        response_quality = np.stack([r["quality"] for r in response], axis=0)
        repsonse_cluster_id = np.array([r["cluster_id"] for r in response])
        replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            mdp_id=0,
            sequence_number=0,
            doc_quality=quality,
            doc_cluster_id=cluster_id,
            response_click=click,
            response_cluster_id=repsonse_cluster_id,
            response_quality=response_quality,
            log_prob=log_prob,
        )

    @unittest.skipIf(not HAS_RECSIM, "recsim is not installed")
    def test_create_from_recsim_interest_evolution(self):
        env = RecSim(num_candidates=20, slate_size=3, resample_documents=False)
        replay_buffer = ReplayBuffer(replay_capacity=100, batch_size=10)
        obs = env.reset()
        observation = obs["user"]
        action = env.action_space.sample()
        log_prob = -1.0
        doc_features = np.stack(list(obs["doc"].values()), axis=0)

        next_obs, reward, terminal, _env = env.step(action)

        response = next_obs["response"]
        click = np.array([r["click"] for r in response])
        response_quality = np.stack([r["quality"] for r in response], axis=0)
        repsonse_cluster_id = np.array([r["cluster_id"] for r in response])
        response_watch_time = np.stack([r["watch_time"] for r in response], axis=0)
        response_liked = np.array([r["liked"] for r in response])
        replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            mdp_id=0,
            sequence_number=0,
            doc=doc_features,
            response_click=click,
            response_cluster_id=repsonse_cluster_id,
            response_quality=response_quality,
            response_liked=response_liked,
            response_watch_time=response_watch_time,
            log_prob=log_prob,
        )
