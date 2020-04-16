#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from gym import Env
from reagent.gym.policies.policy import Policy
from reagent.gym.agents.agent import Agent
from reagent.gym.types import (
    PolicyPreprocessor,
    ActionPreprocessor,
    PostStep,
    Scorer,
    Sampler,
)


def run_episode(
    env: Env,
    policy: Policy,
    action_preprocessor: ActionPreprocessor,
    post_step: PostStep,
    max_steps: Optional[int] = None,
) -> float:
    """
    Constructs the agent and runs it for an episode.
    Return sum of rewards from episode.
    """
    agent = Agent(
        policy=policy, action_preprocessor=action_preprocessor, post_step=post_step
    )

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
