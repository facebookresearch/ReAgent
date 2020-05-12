#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from gym import Env
from reagent.gym.agents.agent import Agent
from reagent.tensorboardX import SummaryWriterContext


def run_episode(env: Env, agent: Agent, max_steps: Optional[int] = None) -> float:
    """
    Return sum of rewards from episode.
    After max_steps (if specified), the environment is assumed to be terminal.
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

        agent.post_step(reward, terminal)
        SummaryWriterContext.increase_global_step()
    return ep_reward
