#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import multiprocessing
import pickle
from typing import List, Optional, Sequence

import numpy as np
from gym import Env
from reagent.core.multiprocess_utils import (
    unwrap_function_outputs,
    wrap_function_arguments,
)
from reagent.gym.agents.agent import Agent
from reagent.gym.types import Trajectory, Transition
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


def run_episode(
    env: Env, agent: Agent, mdp_id: int = 0, max_steps: Optional[int] = None
) -> Trajectory:
    """
    Return sum of rewards from episode.
    After max_steps (if specified), the environment is assumed to be terminal.
    Can also specify the mdp_id and gamma of episode.
    """
    trajectory = Trajectory()
    obs = env.reset()
    terminal = False
    num_steps = 0
    while not terminal:
        action = agent.act(obs)
        next_obs, reward, terminal, _ = env.step(action)
        num_steps += 1
        if max_steps is not None and num_steps > max_steps:
            terminal = True

        # Only partially filled. Agent can fill in more fields.
        transition = Transition(
            mdp_id=mdp_id,
            sequence_number=num_steps,
            observation=obs,
            action=action,
            reward=reward,
            terminal=terminal,
        )
        agent.post_step(transition)
        trajectory.add_transition(transition)
        SummaryWriterContext.increase_global_step()
        obs = next_obs
    return trajectory


def evaluate_for_n_episodes(
    n: int,
    env: Env,
    agent: Agent,
    max_steps: Optional[int] = None,
    gammas: Sequence[float] = (1.0,),
    num_processes: int = 4,
) -> np.ndarray:
    """ Return an np array A of shape n x len(gammas)
        where A[i, j] = ith episode evaluated with gamma=gammas[j].
        Runs environments on num_processes, via multiprocessing.Pool.
    """

    def evaluate_one_episode(
        mdp_id: int,
        env: Env,
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

    def singleprocess_run():
        rewards = []
        for i in range(n):
            rewards.append(
                evaluate_one_episode(
                    mdp_id=i, env=env, agent=agent, max_steps=max_steps, gammas=gammas
                )
            )
        rewards = np.array(rewards)
        return rewards

    if num_processes > 1:
        try:
            with multiprocessing.Pool(num_processes) as pool:
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
                rewards = np.array(rewards)
        except pickle.PickleError as e:
            # NOTE: Probably tried to perform mixed serialization of ScriptModule
            # and non-script modules. This isn't supported right now.
            logger.info(
                f"Trying single processing instead, since got pickle error: {e}."
            )
            rewards = singleprocess_run()
    else:
        rewards = singleprocess_run()

    logger.info(f"Average eval reward is {rewards.mean(axis=0)} for gammas {gammas}.")
    return rewards
