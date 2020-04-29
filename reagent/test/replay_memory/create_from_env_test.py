#!/usr/bin/env python3

import unittest

import numpy as np
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer

try:
    from recsim.environments import interest_exploration
    HAS_RECSIM = True
except ModuleNotFoundError:
    HAS_RECSIM = False


class CreateFromEnvTest(unittest.TestCase):
    @unittest.skipIf(not HAS_RECSIM)
    def test_create_from_recsim(self):
        env_config = {
            "num_candidates": 20,
            "slate_size": 3,
            "resample_documents": False,
            "seed": 1,
        }
        env = interest_exploration.create_environment(env_config)
        replay_buffer = ReplayBuffer.create_from_env(
            env,
            replay_memory_size=100,
            batch_size=10,
            store_possible_actions_mask=False,
            store_log_prob=True,
        )
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
            observation,
            action,
            reward,
            terminal,
            doc_quality=quality,
            doc_cluster_id=cluster_id,
            response_click=click,
            response_cluster_id=repsonse_cluster_id,
            response_quality=response_quality,
            log_prob=log_prob,
        )
