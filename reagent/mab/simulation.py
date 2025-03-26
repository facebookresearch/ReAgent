#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from reagent.mab.mab_algorithm import MABAlgo
from torch import Tensor

# see https://fburl.com/anp/f7y0gzl8 for an example of how evaluate MAB algorithms using a simulation


class MAB(ABC):
    @abstractmethod
    def __init__(
        self,
        max_steps: int,
        expected_rewards: Tensor,
        arm_ids: Optional[List[str]] = None,
    ):
        self.max_steps = max_steps
        self.expected_rewards = expected_rewards
        self.best_action_value = expected_rewards.max().item()
        self.best_action_id = torch.argmax(expected_rewards).item()
        if arm_ids is None:
            self.arm_ids = list(map(str, range(len(expected_rewards))))
        else:
            self.arm_ids = arm_ids
        self.t = 0

    @abstractmethod
    def act(self, arm_id: str) -> float:
        pass

    @property
    def n_arms(self) -> int:
        return len(self.expected_rewards)


class BernoilliMAB(MAB):
    """
    A class that simulates a bandit

    Args:
        probs: A tensor of per-arm success probabilities
        max_steps: Max number os steps to simulate. This has to be specified because we pre-generate
            all the rewards at initialization (for speedup - generating random matrix once should be
            faster than generating random scalars in a loop)
    """

    def __init__(
        self,
        max_steps: int,
        probs: torch.Tensor,
        arm_ids: Optional[List[str]] = None,
    ) -> None:
        """ """
        assert probs.max() <= 1.0
        assert probs.min() >= 0.0
        super().__init__(max_steps=max_steps, expected_rewards=probs, arm_ids=arm_ids)
        self.rewards = torch.bernoulli(
            probs.repeat(max_steps, 1)
        )  # pre-generate all rewards ahead of time
        assert self.rewards.shape == (max_steps, len(probs))

        self.best_action_value = probs.max().item()

    def act(self, arm_id: str) -> float:
        """
        Sample a reward from a specific arm

        Args:
            arm_idx: Index of arm from which reward is sampled
        Returns:
            Sampled reward
        """
        arm_idx = self.arm_ids.index(arm_id)
        assert arm_idx <= (len(self.expected_rewards) - 1)
        assert self.t < self.max_steps
        val = self.rewards[self.t, arm_idx].item()
        self.t += 1
        return val


def single_evaluation_bandit_algo(
    bandit: MAB,
    algo: MABAlgo,
    *,
    update_every: int = 1,
    freeze_scores_btw_updates: bool = True,
) -> np.ndarray:
    """
    Evaluate a bandit algorithm on a single bandit instance.
    Pseudo-regret (difference between expected values of best and chosen actions) is used to minimize variance of evaluation

    Args:
        bandit: Bandit instance on which we evaluate
        algo: Bandit algorithm to be evaluated
        update_every: How many steps between the model is updated. 1 is online learning, >1 is iterative batch learning.
        freeze_scores_btw_updates: If True, the scores are frozen between model updates, otherwise at each step we generate
            new scores even if the model wasn't updated. `False` doesn't make sense for UCB models since the scores are deterministic
            and wouldn't change until the model is updated. Use `False` only for models with non-deterministic scores, like Thompson sampling.
    Returns:
        An array of cumulative pseudo regret
    """
    rewards = []
    expected_rewards = []
    # iterate through model updates
    remaining_steps = bandit.max_steps
    for _ in range(0, bandit.max_steps, update_every):
        batch_n_obs_per_arm = torch.zeros(bandit.n_arms)
        batch_sum_reward_per_arm = torch.zeros(bandit.n_arms)
        batch_sum_squared_reward_per_arm = torch.zeros(bandit.n_arms)
        steps_before_update = min(
            remaining_steps, update_every
        )  # take this many steps until next model update
        arm_id = algo.get_action()  # this action will be reused until next model update if freeze_scores_btw_updates
        for i in range(steps_before_update):
            # iterate through steps without updating the model
            if (not freeze_scores_btw_updates) and (i > 0):
                # if scores are not frozen, we choose new action at each step
                # (except first, because we've already chosen the first action above)
                arm_id = algo.get_action()
            arm_idx = algo.arm_ids.index(arm_id)
            reward = bandit.act(arm_id)
            rewards.append(reward)
            expected_rewards.append(bandit.expected_rewards[arm_idx].item())
            batch_n_obs_per_arm[arm_idx] += 1
            batch_sum_reward_per_arm[arm_idx] += reward
            batch_sum_squared_reward_per_arm[arm_idx] += reward**2
        assert sum(batch_n_obs_per_arm) == steps_before_update
        # perform batch update
        algo.add_batch_observations(
            batch_n_obs_per_arm,
            batch_sum_reward_per_arm,
            batch_sum_squared_reward_per_arm,
        )
        remaining_steps -= steps_before_update
    assert remaining_steps == 0
    assert len(rewards) == bandit.max_steps
    per_step_pseudo_regret = bandit.best_action_value - np.array(expected_rewards)
    return np.cumsum(per_step_pseudo_regret)


def multiple_evaluations_bandit_algo(
    algo_cls: Type[MABAlgo],
    bandit_cls: Type[MAB],
    n_bandits: int,
    max_steps: int,
    update_every: int = 1,
    freeze_scores_btw_updates: bool = True,
    num_processes: Optional[int] = None,
    algo_kwargs: Optional[Dict] = None,
    bandit_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    """
    Perform evaluations on multiple bandit instances and aggregate (average) the result

    Args:
        algo_cls: MAB algorithm class to be evaluated
        bandit_cls: Bandit class on which we perform evaluations
        n_bandits: Number of bandit instances among which the results are averaged
        max_steps: Number of time steps to simulate
        update_every: How many steps between the model is updated. 1 is online learning, >1 is iterative batch learning.
        freeze_scores_btw_updates: If True, the scores are frozen between model updates, otherwise at each step we generate
            new scores even if the model wasn't updated. `False` doesn't make sense for UCB models since the scores are deterministic
            and wouldn't change until the model is updated. Use `False` only for models with non-deterministic scores, like Thompson sampling.
        algo_kwargs: A dict of kwargs to pass to algo_cls at initialization
        bandit_kwargs: A dict of kwargs to pass to bandit_cls at initialization
    Returns:
        An array of cumulative pseudo regret (average across multiple bandit instances)
    """
    if algo_kwargs is None:
        algo_kwargs = {}
    if bandit_kwargs is None:
        bandit_kwargs = {}
    pseudo_regrets = []
    arguments = (
        (
            bandit_cls(max_steps=max_steps, **bandit_kwargs),  # pyre-ignore[45]
            algo_cls(**algo_kwargs),  # pyre-ignore[45]
        )
        for _ in range(n_bandits)
    )
    if num_processes == 1:
        pseudo_regrets = [
            single_evaluation_bandit_algo(
                *a,
                update_every=update_every,
                freeze_scores_btw_updates=freeze_scores_btw_updates,
            )
            for a in arguments
        ]
    else:
        with Pool(num_processes) as pool:
            pseudo_regrets = pool.starmap(
                partial(
                    single_evaluation_bandit_algo,
                    update_every=update_every,
                    freeze_scores_btw_updates=freeze_scores_btw_updates,
                ),
                arguments,
            )
    return np.stack(pseudo_regrets).mean(0)


def compare_bandit_algos(
    algo_clss: List[Type[MABAlgo]],
    bandit_cls: Type[MAB],
    n_bandits: int,
    max_steps: int,
    update_every: int = 1,
    freeze_scores_btw_updates: bool = True,
    algo_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    bandit_kwargs: Optional[Dict] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Args:
        algo_clss: A list of MAB algorithm classes to be evaluated
        bandit_cls: Bandit class on which we perform evaluations
        n_bandits: Number of bandit instances among which the results are averaged
        max_steps: Number of time steps to simulate
        update_every: How many steps between the model is updated. 1 is online learning, >1 is iterative batch learning.
        freeze_scores_btw_updates: If True, the scores are frozen between model updates, otherwise at each step we generate
            new scores even if the model wasn't updated. `False` doesn't make sense for UCB models since the scores are deterministic
            and wouldn't change until the model is updated. Use `False` only for models with non-deterministic scores, like Thompson sampling.
        algo_kwargs: A dict (or list of dicts, one per algorightm class) of kwargs to pass to algo_cls at initialization
        bandit_kwargs: A dict of kwargs to pass to bandit_cls at initialization
    Returns:
        A list of algorithm names that were evaluated (based on class names)
        A list of cumulative regret trajectories (one per evaluated algorithm)
    """
    if algo_kwargs is None:
        algo_kwargs = {}
    if bandit_kwargs is None:
        bandit_kwargs = {}
    if isinstance(algo_kwargs, Dict):
        algo_kwargs = [algo_kwargs] * len(algo_clss)
    names = []
    pseudo_regrets = []
    for algo_cls, algo_kwargs_this_algo in zip(algo_clss, algo_kwargs):
        names.append(algo_cls.__name__)
        pseudo_regrets.append(
            multiple_evaluations_bandit_algo(
                algo_cls=algo_cls,
                bandit_cls=bandit_cls,
                n_bandits=n_bandits,
                max_steps=max_steps,
                update_every=update_every,
                freeze_scores_btw_updates=freeze_scores_btw_updates,
                algo_kwargs=algo_kwargs_this_algo,
                bandit_kwargs=bandit_kwargs,
            )
        )
    return names, pseudo_regrets
