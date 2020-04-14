#!/usr/bin/env python3

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import count, zip_longest
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from reagent.ope.estimators.estimator import Estimator, EstimatorResults
from reagent.ope.estimators.types import (
    Action,
    ActionDistribution,
    ActionSpace,
    Probability,
    Reward,
    TypeWrapper,
)
from reagent.ope.utils import Clamper, RunningAverage
from torch import Tensor


StateType = Union[float, Tuple[float], Tuple[int], np.ndarray, Tensor]


@dataclass(frozen=True)
class State(TypeWrapper[StateType]):
    is_terminal: bool = False

    def __repr__(self):
        return super().__repr__()[:-1] + f",is_terminal[{self.is_terminal}]}}"


@dataclass(frozen=True)
class StateReward:
    state: State = None
    reward: Reward = 0.0


@dataclass(frozen=True)
class RewardProbability:
    reward: Reward = 0.0
    prob: Probability = 0.0


# State distribution: State -> (reward, probability)
StateDistribution = Mapping[State, RewardProbability]


@dataclass(frozen=True)
class Transition:
    class Status(Enum):
        NOOP = 0
        NORMAL = 1
        TERMINATED = 2

    last_state: Optional[State] = None  # from state
    action: Optional[Action] = None  # action
    action_prob: float = 0.0  # action probability
    state: Optional[State] = None  # to state
    reward: float = 0.0
    status: Status = Status.NORMAL


# MDP sequence
Mdp = Sequence[Transition]


class RLPolicy(ABC):
    """
    Policy interface
    """

    def __init__(self, action_space: ActionSpace, device=None):
        self._action_space = action_space
        self._device = device

    @abstractmethod
    def action_dist(self, state: State) -> ActionDistribution:
        pass

    def __call__(self, state: State) -> ActionDistribution:
        return self.action_dist(state)

    @property
    def action_space(self):
        return self._action_space


class RandomRLPolicy(RLPolicy):
    """
    A random policy which return an action according to uniform distribution
    """

    def __init__(self, action_space: ActionSpace, device=None):
        super().__init__(action_space, device)
        self._prob = 1.0 / len(action_space)

    def action_dist(self, state: State) -> ActionDistribution:
        return self._action_space.distribution([self._prob] * len(self._action_space))


class EpsilonGreedyRLPolicy(RLPolicy):
    """
    A wrapper policy:
        Skewing the wrapped policy action distribution by epsilon
        Number of total actions must be given, and wrapped policy should
        calculate probabilities for all actions
    """

    def __init__(self, policy: RLPolicy, epsilon: float = 0.0):
        assert policy is not None and 0 <= epsilon < 1
        super().__init__(policy._device)
        self._policy = policy
        self._exploitation_prob = 1.0 - epsilon
        self._exploration_prob = epsilon / len(policy.action_space)

    def action_dist(self, state) -> ActionDistribution:
        new_dist = self._policy(state).copy()
        for a, p in new_dist:
            new_dist[a] = p * self._exploitation_prob + self._exploration_prob
        return new_dist


class Model(ABC):
    """
    Model interface
    """

    @abstractmethod
    def next_state_reward_dist(self, state: State, action: Action) -> StateDistribution:
        pass

    def __call__(self, state: State, action: Action) -> StateDistribution:
        return self.next_state_reward_dist(state, action)


class ValueFunction(ABC):
    """
    Value function to calculate state and state-action values
    """

    @abstractmethod
    def state_action_value(self, state: State, action: Action) -> float:
        pass

    @abstractmethod
    def state_value(self, state: State) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass

    def __call__(self, state: State, action: Optional[Action] = None) -> float:
        return (
            self.state_action_value(state, action)
            if action is not None
            else self.state_value(state)
        )


@dataclass(frozen=True)
class RLEstimatorInput:
    gamma: float
    log: Mapping[State, Sequence[Mdp]]
    target_policy: RLPolicy
    value_function: Optional[ValueFunction] = None
    ground_truth: Optional[ValueFunction] = None
    horizon: int = -1


class RLEstimator(Estimator):
    def _log_reward(self, gamma: float, mdps: Sequence[Mdp]) -> float:
        avg = RunningAverage()
        for mdp in mdps:
            discount = 1.0
            r = 0.0
            for t in mdp:
                r += discount * t.reward
                discount *= gamma
            avg.add(r)
        return avg.average


class DMEstimator(RLEstimator):
    """
    Direct Method estimator
    """

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        assert input.value_function is not None
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        self.reset()
        for state, mdps in input.log.items():
            estimate = input.value_function(state)
            if input.ground_truth is not None:
                ground_truth = input.ground_truth(state)
            else:
                ground_truth = None
            self._append_estimate(
                self._log_reward(input.gamma, mdps), estimate, ground_truth
            )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return self.results


class IPSEstimator(RLEstimator):
    """
    IPS estimator
    """

    def __init__(
        self, weight_clamper: Clamper = None, weighted: bool = True, device=None
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def _calc_weights(
        self,
        episodes: int,
        horizon: int,
        mdp_transitions: Iterable[Iterable[Transition]],
        policy: RLPolicy,
    ) -> torch.Tensor:
        pi_e = torch.ones((episodes, horizon))
        pi_b = torch.ones((episodes, horizon))
        mask = torch.ones((episodes, horizon))
        j = 0
        for ts in mdp_transitions:
            i = 0
            for t in ts:
                if t is not None and t.action is not None and t.action_prob > 0.0:
                    pi_e[i, j] = policy(t.last_state)[t.action]
                    pi_b[i, j] = t.action_prob
                else:
                    mask[i, j] = 0.0
                i += 1
            j += 1
        pi_e = pi_e.to(device=self._device)
        pi_b = pi_b.to(device=self._device)
        mask = mask.to(device=self._device)
        rho = pi_e.div_(pi_b).cumprod(1).mul_(mask)
        if self._weighted:
            weight = rho.sum(0)
        else:
            weight = mask.sum(0)
        weight.add_(weight.lt(1.0e-15) * episodes)
        ws = rho / weight
        return self._weight_clamper(ws)

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        self.reset()
        for state, mdps in input.log.items():
            n = len(mdps)
            horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, mdps))
            weights = self._calc_weights(
                n, horizon, zip_longest(*mdps), input.target_policy
            )
            discount = torch.full((horizon,), input.gamma, device=self._device)
            discount[0] = 1.0
            discount = discount.cumprod(0)
            rewards = torch.zeros((n, horizon))
            j = 0
            for ts in zip_longest(*mdps):
                i = 0
                for t in ts:
                    if t is not None:
                        rewards[i, j] = t.reward
                    i += 1
                j += 1
            rewards = rewards.to(device=self._device)
            estimate = weights.mul(rewards).sum(0).mul(discount).sum().item()
            if input.ground_truth is not None:
                ground_truth = input.ground_truth(state)
            else:
                ground_truth = None
            self._append_estimate(
                self._log_reward(input.gamma, mdps), estimate, ground_truth
            )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return self.results

    def __repr__(self):
        return super().__repr__()[0:-1] + f",weighted[{self._weighted}]}}"


class DREstimator(IPSEstimator):
    """
    Doubly Robust estimator
    """

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        self.reset()
        for state, mdps in input.log.items():
            n = len(mdps)
            horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, mdps))
            ws = self._calc_weights(n, horizon, zip_longest(*mdps), input.target_policy)
            last_ws = torch.zeros((n, horizon), device=self._device)
            last_ws[:, 0] = 1.0 / n
            last_ws[:, 1:] = ws[:, :-1]
            discount = torch.full((horizon,), input.gamma, device=self._device)
            discount[0] = 1.0
            discount = discount.cumprod(0)
            rs = torch.zeros((n, horizon))
            vs = torch.zeros((n, horizon))
            qs = torch.zeros((n, horizon))
            for ts, j in zip(zip_longest(*mdps), count()):
                for t, i in zip(ts, count()):
                    if t is not None and t.action is not None:
                        qs[i, j] = input.value_function(t.last_state, t.action)
                        vs[i, j] = input.value_function(t.last_state)
                        rs[i, j] = t.reward
            vs = vs.to(device=self._device)
            qs = qs.to(device=self._device)
            rs = rs.to(device=self._device)
            estimate = ((ws * (rs - qs) + last_ws * vs).sum(0) * discount).sum().item()
            if input.ground_truth is not None:
                ground_truth = input.ground_truth(state)
            else:
                ground_truth = None
            self._append_estimate(
                self._log_reward(input.gamma, mdps), estimate, ground_truth
            )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return self.results


class MAGICEstimator(IPSEstimator):
    """
    Algorithm from https://arxiv.org/abs/1604.00923, appendix G.3
    """

    def __init__(self, weight_clamper: Clamper = None, device=None):
        super().__init__(weight_clamper, True, device)

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        assert input.value_function is not None
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        self.reset()
        num_resamples = kwargs["num_resamples"] if "num_resamples" in kwargs else 200
        loss_threhold = (
            kwargs["loss_threhold"] if "loss_threhold" in kwargs else 0.00001
        )
        lr = kwargs["lr"] if "lr" in kwargs else 0.0001
        logging.info(
            f"  params: num_resamples[{num_resamples}], "
            f"loss_threshold[{loss_threhold}], "
            f"lr[{lr}]"
        )
        for state, mdps in input.log.items():
            n = len(mdps)
            horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, mdps))
            ws = self._calc_weights(n, horizon, zip_longest(*mdps), input.target_policy)
            last_ws = torch.zeros((n, horizon), device=self._device)
            last_ws[:, 0] = 1.0 / n
            last_ws[:, 1:] = ws[:, :-1]
            discount = torch.full((horizon,), input.gamma, device=self._device)
            discount[0] = 1.0
            discount = discount.cumprod(0)
            rs = torch.zeros((n, horizon))
            vs = torch.zeros((n, horizon))
            qs = torch.zeros((n, horizon))
            for ts, j in zip(zip_longest(*mdps), count()):
                for t, i in zip(ts, count()):
                    if t is not None and t.action is not None:
                        qs[i, j] = input.value_function(t.last_state, t.action)
                        vs[i, j] = input.value_function(t.last_state)
                        rs[i, j] = t.reward
            vs = vs.to(device=self._device)
            qs = qs.to(device=self._device)
            rs = rs.to(device=self._device)
            wdrs = ((ws * (rs - qs) + last_ws * vs) * discount).cumsum(1)
            wdr = wdrs[:, -1].sum(0)
            next_vs = torch.zeros((n, horizon), device=self._device)
            next_vs[:, :-1] = vs[:, 1:]
            gs = wdrs + ws * next_vs * discount
            gs_normal = gs.sub(torch.mean(gs, 0))
            omiga = n * torch.einsum("ij,ik->jk", gs_normal, gs_normal) / (n - 1.0)
            resample_wdrs = torch.zeros((num_resamples,))
            for i in range(num_resamples):
                samples = random.choices(range(n), k=n)
                sws = ws[samples, :]
                last_sws = last_ws[samples, :]
                srs = rs[samples, :]
                svs = vs[samples, :]
                sqs = qs[samples, :]
                resample_wdrs[i] = (
                    ((sws * (srs - sqs) + last_sws * svs).sum(0) * discount)
                    .sum()
                    .item()
                )
            resample_wdrs, _ = resample_wdrs.to(device=self._device).sort(0)
            lb = torch.min(wdr, resample_wdrs[int(round(0.05 * num_resamples))])
            ub = torch.max(wdr, resample_wdrs[int(round(0.95 * num_resamples)) - 1])
            b = torch.tensor(
                list(
                    map(
                        lambda a: a - ub if a > ub else (a - lb if a < lb else 0.0),
                        gs.sum(0),
                    )
                ),
                device=self._device,
            )
            b.unsqueeze_(0)
            bb = b * b.t()
            cov = omiga + bb
            # x = torch.rand((1, horizon), device=self.device, requires_grad=True)
            x = torch.zeros((1, horizon), device=self._device, requires_grad=True)
            # using SGD to find min x
            optimizer = torch.optim.SGD([x], lr=lr)
            last_y = 0.0
            for i in range(100):
                x = torch.nn.functional.softmax(x, dim=1)
                y = torch.mm(torch.mm(x, cov), x.t())
                if abs(y.item() - last_y) < loss_threhold:
                    print(f"{i}: {last_y} -> {y.item()}")
                    break
                last_y = y.item()
                optimizer.zero_grad()
                y.backward(retain_graph=True)
                optimizer.step()
            x = torch.nn.functional.softmax(x, dim=1)
            estimate = torch.mm(x, gs.sum(0, keepdim=True).t())
            if input.ground_truth is not None:
                ground_truth = input.ground_truth(state)
            else:
                ground_truth = None
            self._append_estimate(
                self._log_reward(input.gamma, mdps), estimate, ground_truth
            )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return self.results
