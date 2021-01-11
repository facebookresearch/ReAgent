#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import asyncio
import logging
import pickle
from typing import Optional, Sequence

import numpy as np
import torch.multiprocessing as mp
from reagent.core.multiprocess_utils import (
    unwrap_function_outputs,
    wrap_function_arguments,
)
from reagent.gym.agents.agent import Agent
from reagent.gym.envs import EnvWrapper
from reagent.gym.types import Trajectory, Transition
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


def run_episode(
    env: EnvWrapper,
    agent: Agent,
    mdp_id: int = 0,
    max_steps: Optional[int] = None,
    fill_info: bool = False,
) -> Trajectory:
    return asyncio.run(
        async_run_episode(
            env=env,
            agent=agent,
            mdp_id=mdp_id,
            max_steps=max_steps,
            fill_info=fill_info,
        )
    )


async def async_run_episode(
    env: EnvWrapper,
    agent: Agent,
    mdp_id: int = 0,
    max_steps: Optional[int] = None,
    fill_info: bool = False,
) -> Trajectory:
    """
    NOTE: this funciton is an async coroutine in order to support async env.step(). If you are using
        it with regular env.step() method, use non-async run_episode(), which wraps this function.
    Return sum of rewards from episode.
    After max_steps (if specified), the environment is assumed to be terminal.
    Can also specify the mdp_id and gamma of episode.
    """
    trajectory = Trajectory()
    obs = env.reset()
    possible_actions_mask = env.possible_actions_mask
    terminal = False
    num_steps = 0
    step_is_coroutine = asyncio.iscoroutinefunction(env.step)
    while not terminal:
        action, log_prob = agent.act(obs, possible_actions_mask)
        if step_is_coroutine:
            next_obs, reward, terminal, info = await env.step(action)
        else:
            next_obs, reward, terminal, info = env.step(action)
        if not fill_info:
            info = None
        next_possible_actions_mask = env.possible_actions_mask
        if max_steps is not None and num_steps >= max_steps:
            terminal = True

        # Only partially filled. Agent can fill in more fields.
        transition = Transition(
            mdp_id=mdp_id,
            sequence_number=num_steps,
            observation=obs,
            action=action,
            reward=float(reward),
            terminal=bool(terminal),
            log_prob=log_prob,
            possible_actions_mask=possible_actions_mask,
            info=info,
        )
        agent.post_step(transition)
        trajectory.add_transition(transition)
        SummaryWriterContext.increase_global_step()
        obs = next_obs
        possible_actions_mask = next_possible_actions_mask
        num_steps += 1
    agent.post_episode(trajectory)
    return trajectory


def evaluate_for_n_episodes(
    n: int,
    env: EnvWrapper,
    agent: Agent,
    max_steps: Optional[int] = None,
    gammas: Sequence[float] = (1.0,),
    num_processes: int = 4,
) -> np.ndarray:
    """Return an np array A of shape n x len(gammas)
    where A[i, j] = ith episode evaluated with gamma=gammas[j].
    Runs environments on num_processes, via multiprocessing.Pool.
    """
    num_processes = min(num_processes, n)

    def evaluate_one_episode(
        mdp_id: int,
        env: EnvWrapper,
        agent: Agent,
        max_steps: Optional[int],
        gammas: Sequence[float],
    ) -> np.ndarray:
        rewards = np.empty((len(gammas),))
        trajectory = run_episode(
            env=env, agent=agent, mdp_id=mdp_id, max_steps=max_steps
        )
        for i_gamma, gamma in enumerate(gammas):
            rewards[i_gamma] = trajectory.calculate_cumulative_reward(gamma)
        return rewards

    rewards = None
    if num_processes > 1:
        try:
            with mp.Pool(num_processes) as pool:
                rewards = unwrap_function_outputs(
                    pool.map(
                        wrap_function_arguments(
                            evaluate_one_episode,
                            env=env,
                            agent=agent,
                            max_steps=max_steps,
                            gammas=gammas,
                        ),
                        range(n),
                    )
                )
        except pickle.PickleError as e:
            # NOTE: Probably tried to perform mixed serialization of ScriptModule
            # and non-script modules. This isn't supported right now.
            logger.info(e)
            logger.info(
                "This is probably from trying to serialize a TorchScript module, "
                "wrapped in a non-script module. Mixed serialization is not supported."
            )

    # if we didn't run multiprocessing, or it failed, try single-processing instead.
    if rewards is None:
        rewards = []
        for i in range(n):
            rewards.append(
                evaluate_one_episode(
                    mdp_id=i, env=env, agent=agent, max_steps=max_steps, gammas=gammas
                )
            )

    rewards = np.array(rewards)
    for i, gamma in enumerate(gammas):
        gamma_rewards = rewards[:, i]
        logger.info(
            f"For gamma={gamma}, average reward is {gamma_rewards.mean()}\n"
            f"Rewards list: {gamma_rewards}"
        )
    return rewards
