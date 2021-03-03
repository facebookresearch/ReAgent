#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import pickle
from typing import Optional, Sequence, Tuple

import numpy as np
import torch.multiprocessing as mp
from reagent.core.multiprocess_utils import (
    unwrap_function_outputs,
    wrap_function_arguments,
)
from reagent.gym.agents.agent import Agent
from reagent.gym.envs import EnvWrapper
from reagent.gym.types import Trajectory, Transition, EvaluationResults
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


def run_episode(
    env: EnvWrapper, agent: Agent, mdp_id: int = 0, max_steps: Optional[int] = None
) -> Trajectory:
    """
    Return sum of rewards from episode.
    After max_steps (if specified), the environment is assumed to be terminal.
    Can also specify the mdp_id and gamma of episode.
    """
    trajectory = Trajectory()
    obs = env.reset()
    possible_actions_mask = env.possible_actions_mask
    terminal = False
    num_steps = 0
    while not terminal:
        action, log_prob = agent.act(obs, possible_actions_mask)
        next_obs, reward, terminal, _ = env.step(action)
        next_possible_actions_mask = env.possible_actions_mask
        if max_steps is not None and num_steps >= (max_steps - 1):
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
) -> EvaluationResults:
    """Return an np array A of shape n x len(gammas)
    where A[i, j] = ith episode evaluated with gamma=gammas[j].
    Runs environments on num_processes, via multiprocessing.Pool.
    """
    num_processes = min(num_processes, n)

    metric_extractor = env.get_metric_extractor()

    def evaluate_one_episode(
        mdp_id: int,
        env: EnvWrapper,
        agent: Agent,
        max_steps: Optional[int],
        gammas: Sequence[float],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rewards = np.empty((len(gammas),))
        trajectory = run_episode(
            env=env, agent=agent, mdp_id=mdp_id, max_steps=max_steps
        )
        for i_gamma, gamma in enumerate(gammas):
            rewards[i_gamma] = trajectory.calculate_cumulative_reward(gamma)

        metrics = None

        if metric_extractor is not None:
            metrics = metric_extractor(trajectory)

        return rewards, metrics

    eval_results = None
    if num_processes > 1:
        try:
            with mp.Pool(num_processes) as pool:
                eval_results = unwrap_function_outputs(
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
    if eval_results is None:
        eval_results = []
        for i in range(n):
            eval_results.append(
                evaluate_one_episode(
                    mdp_id=i, env=env, agent=agent, max_steps=max_steps, gammas=gammas
                )
            )

    rewards = np.array([r[0] for r in eval_results])
    for i, gamma in enumerate(gammas):
        gamma_rewards = rewards[:, i]
        logger.info(
            f"For gamma={gamma}, average reward is {gamma_rewards.mean()}\n"
            f"Rewards list: {gamma_rewards}"
        )

    metrics = None
    metric_names = None
    if metric_extractor is not None:
        metrics = np.stack([r[1] for r in eval_results])
        metric_names = metric_extractor.metric_names

        logger.info(
            f"Metric names: {metric_names}; avg: {np.mean(metrics, axis=0)} stddev: {np.std(metrics, axis=0)}"
        )

    return EvaluationResults(
        rewards=rewards, metrics=metrics, metric_names=metric_names
    )
