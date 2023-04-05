#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import uuid
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from reagent.evaluation.cb.synthetic_contextual_bandit_data import (
    DynamicBanditAgent,
    DynamicBanditEnv,
)


def identity_collate(batch):
    assert isinstance(batch, list) and len(batch) == 1, f"Got {batch}"
    return batch[0]


class BanditDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        env: DynamicBanditEnv,
        agent: DynamicBanditAgent,
        num_obs: int,
    ):
        self.env = env
        self.agent = agent
        self.num_obs = num_obs

    def __iter__(self):
        for _ in range(self.num_obs):
            obs = self.env.get_batch()  # obs is a batch
            action, log_prob = self.agent.act(obs=obs)
            obs = self.env.add_chosen_action_reward(chosen_action_idx=action, batch=obs)
            yield obs

    def __len__(self):
        return self.num_obs


def run_dynamic_bandit_env(
    num_unique_batches: int,
    batch_size: int,
    num_arms_per_episode: int,
    feature_dim: int,
    max_epochs: int,
    num_obs: int,
    gpus: int = 0,
    rand_seed: int = 937162211,
) -> Tuple[DynamicBanditAgent, List[float], List[float]]:
    seed_everything(rand_seed)
    pl_trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=int(gpus),
        default_root_dir=f"lightning_log_{str(uuid.uuid4())}",
    )

    agent = DynamicBanditAgent.make_agent(feature_dim=feature_dim)
    env = DynamicBanditEnv(
        num_unique_batches=num_unique_batches,
        batch_size=batch_size,
        num_arms_per_episode=num_arms_per_episode,
        feature_dim=feature_dim,
    )

    dataset = BanditDataset(env=env, agent=agent, num_obs=num_obs)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=identity_collate)
    pl_trainer.fit(agent.trainer, data_loader)
    return agent, env.accumulated_rewards, env.accumulated_regrets


"""
Example run:
    run_dynamic_bandit_env(
        feature_dim = 3,
        num_unique_batches = 10,
        batch_size = 4,
        num_arms_per_episode = 2,
        num_obs = 101,
        max_epochs=1,
        )
"""
