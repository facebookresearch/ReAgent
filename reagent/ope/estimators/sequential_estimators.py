#!/usr/bin/env python3

import copy
import logging
import random
import time
import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import count, zip_longest
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from reagent.ope.estimators.estimator import (
    Estimator,
    EstimatorResult,
    EstimatorResults,
)
from reagent.ope.estimators.types import (
    Action,
    ActionDistribution,
    ActionSpace,
    Probability,
    Reward,
    TypeWrapper,
)
from reagent.ope.trainers.linear_trainers import LinearNet
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
    state: Optional[State] = None
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
        assert policy is not None and 0.0 <= epsilon < 1.0
        super().__init__(policy._device)
        self._policy = policy
        self._exploitation_prob = 1.0 - epsilon
        self._exploration_prob = epsilon / len(policy.action_space)

    def action_dist(self, state) -> ActionDistribution:
        new_dist = deepcopy(self._policy(state))
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
    log: Sequence[Mdp]
    target_policy: RLPolicy
    value_function: Optional[ValueFunction] = None
    ground_truth: Optional[ValueFunction] = None
    horizon: int = -1
    discrete_states: bool = True


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

    def _estimate_value(
        self, gamma: float, mdps: Sequence[Mdp], value_function: ValueFunction
    ) -> float:
        avg = RunningAverage()
        for mdp in mdps:
            discount = 1.0
            r = 0.0
            for t in mdp:
                if t.last_state is None:
                    break
                r += discount * value_function(t.last_state)
                discount *= gamma
            avg.add(r)
        return avg.average


class DMEstimator(RLEstimator):
    """
    Direct Method estimator
    """

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        # kwargs is part of the function signature, so to satisfy pyre it must be included
        assert input.value_function is not None
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        results = EstimatorResults()

        estimate = self._estimate_value(input.gamma, input.log, input.value_function)
        if input.ground_truth is not None:
            gt = self._estimate_value(input.gamma, input.log, input.ground_truth)
        results.append(
            EstimatorResult(
                self._log_reward(input.gamma, input.log),
                estimate,
                None if input.ground_truth is None else gt,
            )
        )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return results


class IPSEstimator(RLEstimator):
    """
    IPS estimator
    """

    def __init__(
        self,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = True,
        device=None,
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
                    assert t.last_state is not None
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
        # kwargs is part of the function signature, so to satisfy pyre it must be included
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        results = EstimatorResults()

        n = len(input.log)
        horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, input.log))
        weights = self._calc_weights(
            n, horizon, zip_longest(*input.log), input.target_policy
        )
        discount = torch.full((horizon,), input.gamma, device=self._device)
        discount[0] = 1.0
        discount = discount.cumprod(0)
        rewards = torch.zeros((n, horizon))
        j = 0
        for ts in zip_longest(*input.log):
            i = 0
            for t in ts:
                if t is not None:
                    rewards[i, j] = t.reward
                i += 1
            j += 1
        rewards = rewards.to(device=self._device)
        estimate = weights.mul(rewards).sum(0).mul(discount).sum().item()

        results.append(
            EstimatorResult(
                self._log_reward(input.gamma, input.log),
                estimate,
                None
                if input.ground_truth is None
                else self._estimate_value(input.gamma, input.log, input.ground_truth),
            )
        )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return results

    def __repr__(self):
        return super().__repr__()[0:-1] + f",weighted[{self._weighted}]}}"


class DoublyRobustEstimator(IPSEstimator):
    """
    Doubly Robust estimator
    """

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        # kwargs is part of the function signature, so to satisfy pyre it must be included
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        results = EstimatorResults()

        n = len(input.log)
        horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, input.log))
        ws = self._calc_weights(
            n, horizon, zip_longest(*input.log), input.target_policy
        )
        last_ws = torch.zeros((n, horizon), device=self._device)
        last_ws[:, 0] = 1.0 / n
        last_ws[:, 1:] = ws[:, :-1]
        discount = torch.full((horizon,), input.gamma, device=self._device)
        discount[0] = 1.0
        discount = discount.cumprod(0)
        rs = torch.zeros((n, horizon))
        vs = torch.zeros((n, horizon))
        qs = torch.zeros((n, horizon))
        for ts, j in zip(zip_longest(*input.log), count()):
            for t, i in zip(ts, count()):
                if t is not None and t.action is not None:
                    assert input.value_function is not None
                    qs[i, j] = input.value_function(t.last_state, t.action)
                    vs[i, j] = input.value_function(t.last_state)
                    rs[i, j] = t.reward
        vs = vs.to(device=self._device)
        qs = qs.to(device=self._device)
        rs = rs.to(device=self._device)
        estimate = ((ws * (rs - qs) + last_ws * vs).sum(0) * discount).sum().item()
        results.append(
            EstimatorResult(
                self._log_reward(input.gamma, input.log),
                estimate,
                None
                if input.ground_truth is None
                else self._estimate_value(input.gamma, input.log, input.ground_truth),
            )
        )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return results


class MAGICEstimator(IPSEstimator):
    """
    Algorithm from https://arxiv.org/abs/1604.00923, appendix G.3
    """

    def __init__(self, weight_clamper: Optional[Clamper] = None, device=None):
        super().__init__(weight_clamper, True, device)

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        assert input.value_function is not None
        logging.info(f"{self}: start evaluating")
        stime = time.process_time()
        results = EstimatorResults()
        num_resamples = kwargs["num_resamples"] if "num_resamples" in kwargs else 200
        loss_threshold = (
            kwargs["loss_threshold"] if "loss_threshold" in kwargs else 0.00001
        )
        lr = kwargs["lr"] if "lr" in kwargs else 0.0001
        logging.info(
            f"  params: num_resamples[{num_resamples}], "
            f"loss_threshold[{loss_threshold}], "
            f"lr[{lr}]"
        )
        # Compute MAGIC estimate
        n = len(input.log)
        horizon = len(reduce(lambda a, b: a if len(a) > len(b) else b, input.log))
        ws = self._calc_weights(
            n, horizon, zip_longest(*input.log), input.target_policy
        )
        last_ws = torch.zeros((n, horizon), device=self._device)
        last_ws[:, 0] = 1.0 / n
        last_ws[:, 1:] = ws[:, :-1]
        discount = torch.full((horizon,), input.gamma, device=self._device)
        discount[0] = 1.0
        discount = discount.cumprod(0)
        rs = torch.zeros((n, horizon))
        vs = torch.zeros((n, horizon))
        qs = torch.zeros((n, horizon))
        for ts, j in zip(zip_longest(*input.log), count()):
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
        assert n > 1
        omiga = (n / (n - 1.0)) * torch.einsum("ij,ik->jk", gs_normal, gs_normal)
        resample_wdrs = torch.zeros((num_resamples,))
        for i in range(num_resamples):
            samples = random.choices(range(n), k=n)
            sws = ws[samples, :]
            last_sws = last_ws[samples, :]
            srs = rs[samples, :]
            svs = vs[samples, :]
            sqs = qs[samples, :]
            resample_wdrs[i] = (
                ((sws * (srs - sqs) + last_sws * svs).sum(0) * discount).sum().item()
            )
        resample_wdrs, _ = resample_wdrs.to(device=self._device).sort(0)
        lb = torch.min(wdr, resample_wdrs[int(round(0.05 * num_resamples))])
        ub = torch.max(wdr, resample_wdrs[int(round(0.95 * num_resamples)) - 1])
        b = torch.tensor(
            list(
                map(
                    lambda a: a - ub if a > ub else (a - lb if a < lb else 0.0),
                    # pyre-fixme[6]: Expected `Iterable[Variable[_T1]]` for 2nd
                    #  param but got `Tensor`.
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
            if abs(y.item() - last_y) < loss_threshold:
                print(f"{i}: {last_y} -> {y.item()}")
                break
            last_y = y.item()
            optimizer.zero_grad()
            y.backward(retain_graph=True)
            optimizer.step()
        x = torch.nn.functional.softmax(x, dim=1)
        estimate = torch.mm(x, gs.sum(0, keepdim=True).t()).cpu().item()

        results.append(
            EstimatorResult(
                self._log_reward(input.gamma, input.log),
                estimate,
                None
                if input.ground_truth is None
                else self._estimate_value(input.gamma, input.log, input.ground_truth),
            )
        )
        logging.info(
            f"{self}: finishing evaluating["
            f"process_time={time.process_time() - stime}]"
        )
        return results


@dataclass
class NeuralDualDICE(RLEstimator):
    # See https://arxiv.org/pdf/1906.04733.pdf sections 4, 5, A
    # Google's implementation: https://github.com/google-research/google-research/tree/master/dual_dice
    """
    Args:
        state_dim: The dimensionality of the state vectors
        action_dim: The number of discrete actions
        deterministic_env: Whether or not the environment is determinstic.
                            Can help with stability of training.
        average_next_v: Whether or not to average the next nu value over all
                        possible actions. Can help with stability of training.
        polynomial_degree: The degree of the convex function f(x) = 1/p * |x|^p
        value_lr: The learning rate for nu
        zeta_lr: The learning rate for zeta
        hidden_dim: The dimensionality of the hidden layers for zeta and v
        hidden_layers: The number of hidden layers for zeta and v
        activation: The activation function for zeta and v
        training_samples: The number of batches to train zeta and v for
        batch_size: The number of samples in each batch
        loss_callback_fn: A function that will be called every reporting_frequency batches,
                            giving the average zeta loss, average nu loss, and self
        reporting_frequency: The number of batches between outputting the state of the training
    """
    state_dim: int
    action_dim: int
    deterministic_env: bool
    average_next_v: bool = False
    polynomial_degree: float = 1.5
    value_lr: float = 0.01
    zeta_lr: float = 0.01
    hidden_dim: int = 64
    hidden_layers: int = 2
    activation = torch.nn.Tanh
    training_samples: int = 100000
    batch_size: int = 2048
    device: typing.Any = None
    loss_callback_fn: Optional[Callable[[float, float, RLEstimator], None]] = None
    reporting_frequency: int = 1000
    # These are initialized in __post_init__() and calms Pyre
    v: typing.Any = None
    zeta: typing.Any = None
    f: typing.Any = None
    fconjugate: typing.Any = None
    zeta_net: typing.Any = None
    v_net: typing.Any = None

    def __post_init__(self):
        conjugate_exponent = self.polynomial_degree / (self.polynomial_degree - 1)
        self.f = self._get_convex_f(self.polynomial_degree)
        self.fconjugate = self._get_convex_f(conjugate_exponent)
        self.reset()

    def _get_convex_f(self, degree):
        return lambda x: (torch.abs(x) ** degree) / degree

    @torch.no_grad()
    def _mdps_value(self, mdps: Sequence[Mdp], gamma: float) -> float:
        self.zeta_net.eval()
        avg = RunningAverage()

        for mdp in mdps:
            discount = 1.0
            r = 0.0
            for t in mdp:
                assert t.last_state is not None, "Expected last_state, got None"
                assert t.action is not None, "Expected action, got None"
                zeta = self.zeta(
                    torch.tensor(t.last_state.value, dtype=torch.float)
                    .reshape(-1, self.state_dim)
                    .to(self.device),
                    torch.nn.functional.one_hot(
                        torch.tensor(t.action.value, dtype=torch.long), self.action_dim
                    )
                    .reshape(-1, self.action_dim)
                    .float()
                    .to(self.device),
                )
                r += discount * t.reward * zeta.cpu().item()
                discount *= gamma
            avg.add(r)
        self.zeta_net.train()
        return avg.average

    @torch.no_grad()
    def _compute_estimates(self, input: RLEstimatorInput) -> EstimatorResults:
        results = EstimatorResults()
        estimate = self._mdps_value(input.log, input.gamma)
        results.append(
            EstimatorResult(
                self._log_reward(input.gamma, input.log),
                estimate,
                None
                if input.ground_truth is None
                else self._estimate_value(input.gamma, input.log, input.ground_truth),
            )
        )
        return results

    def _compute_average_v(self, transition):
        next_vs = [
            transition["tgt_action_props"][:, a].reshape(-1, 1)
            * self.v(
                transition["state"],
                torch.nn.functional.one_hot(
                    torch.tensor(a, dtype=torch.long), self.action_dim
                )
                .reshape(1, -1)
                .float()
                .to(self.device)
                .repeat(transition["state"].shape[0], 1),
            )
            for a in range(self.action_dim)
        ]
        return sum(next_vs)

    def _compute_loss(
        self, gamma: float, transition: Dict, compute_determ_v_loss: bool
    ):
        if self.average_next_v:
            next_v = self._compute_average_v(transition)
        else:
            next_v = self.v(transition["state"], transition["next_action"])
        delta_v = (
            self.v(transition["last_state"], transition["log_action"]) - gamma * next_v
        )
        init_v = self.v(transition["init_state"], transition["init_action"])
        if compute_determ_v_loss:
            unweighted_loss = self.f(delta_v) - (1 - gamma) * init_v
        else:
            zeta = self.zeta(transition["last_state"], transition["log_action"])
            unweighted_loss = (
                delta_v * zeta - self.fconjugate(zeta) - (1 - gamma) * init_v
            )
        weights = torch.full(
            (unweighted_loss.shape[0], 1), gamma, dtype=torch.float
        ).to(device=self.device) ** transition["timestep"].reshape((-1, 1))
        return torch.sum(weights * unweighted_loss) / torch.sum(weights)

    def reset(self):
        self.v_net = LinearNet(
            self.state_dim + self.action_dim,
            self.hidden_dim,
            1,
            self.hidden_layers,
            self.activation,
        )
        self.zeta_net = copy.deepcopy(self.v_net)
        self.v_net.to(self.device)
        self.zeta_net.to(self.device)

        self.v = self._build_function(self.v_net)
        self.zeta = self._build_function(self.zeta_net)

    def _build_function(self, net: torch.nn.Module):
        return lambda s, a: net(torch.cat((s, a), dim=1))

    def _collect_data(self, input: RLEstimatorInput):
        samples = {
            "init_state": [],
            "init_action": [],
            "last_state": [],
            "state": [],
            "log_action": [],
            "next_action": [],
            "tgt_action_props": [],
            "timestep": [],
            "reward": [],
        }
        for mdp in input.log:
            state = mdp[0].last_state
            assert state is not None, "Expected initial state, got None"
            tgt_init_action = input.target_policy.action_dist(state).sample()[0]
            for i, t in enumerate(mdp):
                assert (
                    t.state is not None
                    and t.last_state is not None
                    and t.action is not None
                ), "Expected all fields to be present"
                tgt_dist = input.target_policy.action_dist(t.state)
                tgt_action = tgt_dist.sample()[0]
                samples["init_state"].append(
                    state.value.cpu().numpy()
                    if isinstance(state.value, torch.Tensor)
                    else state.value
                )
                samples["init_action"].append(
                    torch.nn.functional.one_hot(
                        torch.tensor(tgt_init_action.value, dtype=torch.long),
                        self.action_dim,
                    ).float()
                )
                samples["last_state"].append(
                    t.last_state.value.cpu().numpy()
                    if isinstance(t.last_state.value, torch.Tensor)
                    else t.last_state.value
                )
                samples["state"].append(
                    t.state.value.cpu().numpy()
                    if isinstance(t.state.value, torch.Tensor)
                    else t.state.value
                )
                samples["log_action"].append(
                    torch.nn.functional.one_hot(
                        torch.tensor(t.action.value, dtype=torch.long), self.action_dim
                    ).float()
                )
                samples["next_action"].append(
                    torch.nn.functional.one_hot(
                        torch.tensor(tgt_action.value, dtype=torch.long),
                        self.action_dim,
                    ).float()
                )
                samples["tgt_action_props"].append(tgt_dist.values)
                samples["timestep"].append(i)
                samples["reward"].append(t.reward)

        return {
            k: torch.stack(v).to(self.device)
            if "action" in k
            else torch.tensor(v, dtype=torch.float).to(self.device)
            for k, v in samples.items()
        }

    def _sample_batch(self, dataset):
        idxs = np.random.choice(dataset["init_state"].shape[0], self.batch_size)
        return {k: v[idxs] for k, v in dataset.items()}

    def evaluate(self, input: RLEstimatorInput, **kwargs) -> EstimatorResults:
        stime = time.process_time()
        dataset = self._collect_data(input)
        logging.info(f"Data loading time: {time.process_time() - stime}")

        zeta_optim = torch.optim.Adam(self.zeta_net.parameters(), lr=self.zeta_lr)
        v_optim = torch.optim.Adam(self.v_net.parameters(), lr=self.value_lr)
        avg_zeta_loss = RunningAverage()
        avg_v_loss = RunningAverage()
        sample_time = time.process_time()
        for sampled in range(self.training_samples):
            sample = self._sample_batch(dataset)

            zeta_loss = -(self._compute_loss(input.gamma, sample, False))
            # Populate zeta gradients and optimize
            zeta_optim.zero_grad()
            zeta_loss.backward()
            zeta_optim.step()

            if self.deterministic_env:
                v_loss = self._compute_loss(input.gamma, sample, True)
            else:
                v_loss = self._compute_loss(*sample)
            # Populate value gradients and optimize
            v_optim.zero_grad()
            v_loss.backward()
            v_optim.step()

            avg_zeta_loss.add(zeta_loss.cpu().item())
            avg_v_loss.add(v_loss.cpu().item())
            if sampled % self.reporting_frequency == 0:
                report_time = time.process_time() - sample_time
                callback_time = None
                if self.loss_callback_fn is not None:
                    # Pyre gets angry if we don't make callback local
                    callback = self.loss_callback_fn
                    assert callback is not None
                    stime = time.process_time()
                    callback(avg_zeta_loss.average, avg_v_loss.average, self)
                    callback_time = abs(time.process_time() - stime)
                logging.info(
                    f"Samples {sampled}, "
                    f"Avg Zeta Loss {avg_zeta_loss.average}, "
                    f"Avg Value Loss {avg_v_loss.average},\n"
                    f"Time per {self.reporting_frequency} samples: {report_time}"
                    + (
                        ""
                        if callback_time is None
                        else f", Time for callback: {callback_time}"
                    )
                )
                avg_zeta_loss = RunningAverage()
                avg_v_loss = RunningAverage()
                sample_time = time.process_time()
        return self._compute_estimates(input)
