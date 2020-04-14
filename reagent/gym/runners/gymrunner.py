#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from gym import Env
from ml.rl.gym.agents.agent import Agent


def run_episode(env: Env, agent: Agent, max_steps: Optional[int] = None) -> float:
    """
    Return total reward
    """
    ep_reward = 0.0
    obs = env.reset()
    terminal = False
    num_steps = 0
    while not terminal:
        action = agent.act(obs)
        next_obs, reward, terminal, _ = env.step(action)
        obs = next_obs
        ep_reward += reward
        num_steps += 1
        if max_steps is not None and num_steps > max_steps:
            terminal = True

        # add sample to replay buffer and optionally train
        agent.post_step(reward, terminal)
    return ep_reward
