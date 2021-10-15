from abc import ABC
from multiprocessing import Pool
from typing import Type, Optional, Dict, List, Union, Tuple

import numpy as np
import torch
from reagent.mab.mab_algorithm import MABAlgo

# see https://fburl.com/anp/f7y0gzl8 for an example of how evaluate MAB algorithms using a simulation


class MAB(ABC):
    pass


class BernoilliMAB(MAB):
    """
    A class that simulates a bandit

    Args:
        probs: A tensor of per-arm success probabilities
        max_steps: Max number os steps to simulate. This has to be specified because we pre-generate
            all the rewards at initialization
    """

    def __init__(self, max_steps: int, probs: torch.Tensor):
        """ """
        assert probs.max() <= 1.0
        assert probs.min() >= 0.0
        self.probs = probs
        self.max_steps = max_steps
        self.rewards = torch.bernoulli(
            self.probs.repeat(max_steps, 1)
        )  # pre-generate all rewards ahead of time
        assert self.rewards.shape == (max_steps, len(self.probs))
        self.t = 0
        self.best_action_id = torch.argmax(probs).item()
        self.best_action_value = probs.max().item()

    def act(self, arm_idx: int) -> float:
        """
        Sample a reward from a specific arm

        Args:
            arm_idx: Index of arm from which reward is sampled
        Returns:
            Sampled reward
        """
        assert arm_idx <= (len(self.probs) - 1)
        assert self.t < self.max_steps
        val = self.rewards[self.t, arm_idx].item()
        self.t += 1
        return val


def single_evaluation_bandit_algo(bandit: MAB, algo: MABAlgo) -> np.ndarray:
    """
    Evaluate a bandit algorithm on a single bandit instance.
    Pseudo-regret (difference between expected values of best and chosen actions) is used to minimize variance of evaluation

    Args:
        bandit: Bandit instance on which we evaluate
        algo: Bandit algorithm to be evaluated
    Returns:
        An array of cumulative presudo regret
    """
    rewards = []
    expected_rewards = []
    for _ in range(bandit.max_steps):
        arm_id = algo.get_action()
        reward = bandit.act(arm_id)
        algo.add_single_observation(arm_id, reward)
        rewards.append(reward)
        expected_rewards.append(bandit.probs[arm_id].item())
    per_step_pseudo_regret = bandit.best_action_value - np.array(expected_rewards)
    return np.cumsum(per_step_pseudo_regret)


def multiple_evaluations_bandit_algo(
    algo_cls: Type[MABAlgo],
    bandit_cls: Type[MAB],
    n_bandits: int,
    max_steps: int,
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
        algo_kwargs: A dict of kwargs to pass to algo_cls at initialization
        bandit_kwargs: A dict of kwargs to pass to bandit_cls at initialization
    Returns:
        An array of cumulative presudo regret (average across multple bandit instances)
    """
    if algo_kwargs is None:
        algo_kwargs = {}
    if bandit_kwargs is None:
        bandit_kwargs = {}
    pseudo_regrets = []
    arguments = (
        (bandit_cls(max_steps=max_steps, **bandit_kwargs), algo_cls(**algo_kwargs))
        for _ in range(n_bandits)
    )
    with Pool(num_processes) as pool:
        pseudo_regrets = pool.starmap(single_evaluation_bandit_algo, arguments)
    return np.stack(pseudo_regrets).mean(0)


def compare_bandit_algos(
    algo_clss: List[Type[MABAlgo]],
    bandit_cls: Type[MAB],
    n_bandits: int,
    max_steps: int,
    algo_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    bandit_kwargs: Optional[Dict] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Args:
        algo_clss: A list of MAB algorithm classes to be evaluated
        bandit_cls: Bandit class on which we perform evaluations
        n_bandits: Number of bandit instances among which the results are averaged
        max_steps: Number of time steps to simulate
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
                algo_kwargs=algo_kwargs_this_algo,
                bandit_kwargs=bandit_kwargs,
            )
        )
    return names, pseudo_regrets
