#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import heapq
import logging
from collections import defaultdict, deque
from math import floor
from typing import Callable, Dict, Tuple, Optional, List, Any

import nevergrad as ng
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nevergrad.parametrization.choice import Choice

logger = logging.getLogger(__name__)

ANNEAL_RATE = 0.9997
LEARNING_RATE = 0.001
BATCH_SIZE = 512
# People rarely need more than that
MAX_NUM_BEST_SOLUTIONS = 1000
GREEDY_TEMP = 0.0001


def sample_from_logits(
    keyed_logits: Dict[str, nn.Parameter], batch_size: int, temp: float
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Return sampled solutions and sampled log probabilities"""
    sampled_log_probs = torch.zeros(batch_size, 1)
    sampled_solutions = {}
    for k, logits in keyed_logits.items():
        softmax_val = F.softmax(logits / temp, dim=-1).squeeze(0)
        samples = torch.multinomial(softmax_val, batch_size, replacement=True)
        sampled_prob = softmax_val[samples].reshape(-1, 1)
        sampled_log_probs += torch.log(sampled_prob)
        sampled_solutions[k] = samples
    return sampled_solutions, sampled_log_probs


def obj_func_scaler(
    obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]],
    exp_offset_and_scale: Optional[Tuple[float, float]],
) -> Optional[Callable]:
    """
    Scale objective functions to make optimizers get out of local minima more easily.

    The scaling formula is: exp((reward - offset) / scale)

    if obj_exp_offset_scale is None, do not scale the obj_function (i.e., reward == scaled_reward)
    """
    if obj_func is None:
        return None

    if exp_offset_and_scale is not None:
        offset, scale = exp_offset_and_scale

    def obj_func_scaled(*args, **kwargs):
        x = obj_func(*args, **kwargs)
        if exp_offset_and_scale is not None:
            return x, torch.exp((x - offset) / scale)
        else:
            return x, x

    return obj_func_scaled


def _num_of_params(model: nn.Module) -> int:
    return len(torch.cat([p.flatten() for p in model.parameters()]))


def sol_to_tensors(
    sampled_sol: Dict[str, torch.Tensor], input_param: ng.p.Dict
) -> torch.Tensor:
    one_hot = [
        # pyre-fixme[16]: `Parameter` has no attribute `choices`.
        F.one_hot(sampled_sol[k], num_classes=len(input_param[k].choices)).type(
            torch.FloatTensor
        )
        for k in sorted(sampled_sol.keys())
    ]
    batch_tensors = torch.cat(one_hot, dim=-1)
    return batch_tensors


class BestResultsQueue:
    """Maintain the `max_len` lowest numbers"""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.reward_sol_dict = defaultdict(set)
        self.heap = []

    def insert(self, reward: torch.Tensor, sol: Dict[str, torch.Tensor]) -> None:
        # Negate the reward because maximal N elements will be kept
        # in heap, while all optimizers are a minimizer.
        reward = -reward
        sol_str = str(sol)
        # skip duplicated solution
        if reward in self.reward_sol_dict and sol_str in self.reward_sol_dict[reward]:
            return
        self.reward_sol_dict[reward].add(sol_str)
        if len(self.heap) < self.max_len:
            heapq.heappush(self.heap, (reward, sol_str, sol))
        else:
            old_r, old_sol_str, old_sol = heapq.heappushpop(
                self.heap, (reward, sol_str, sol)
            )
            self.reward_sol_dict[old_r].remove(old_sol_str)

    def topk(self, k: int) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        k = min(k, len(self.heap))
        res = heapq.nlargest(k, self.heap)
        # a list of (reward, sol) tuples
        return [(-r[0], r[2]) for r in res]


class ComboOptimizerBase:
    """
    The class contains a series of API to be shared between various combonatorial optimization
    optimizers.

    Basic usage:
    1. Create a parameter space and obj function to be minimized
    2. Create optimizer = SomeComboOptimizer(param, obj_func, ...)
    3. Call optimizer.optimize_step() until the budget exhausts

    optimize_step() encapsulates two main steps:
    a. sample_internal(), which samples promising solutions to query during training.
    b. update_params(), which updates the optimizer's parameters using the rewards obtained
        on the sampled solutions from sample_internal()

    The user is free to manually calling sample_internal() and update_params() separately
    instead of calling optimize_step(). While calling optimize_step() is more succinct in
    code, calling sample_internal() and update_params() separately allows more flexibility
    (e.g., the user may perform any additional customized logic between the two functions).

    Once the training is done (i.e., the user no longer has the budget to call optimize_step()),
    the user can use optimizer.sample() to sample solutions based on the learned optimizer.
    The user can also use optimizer.best_solutions() to return the top best solutions discovered
    during the training.

    Each optimizer has its own doc string test for further reference.
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        batch_size: int = BATCH_SIZE,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        for k in param:
            assert isinstance(
                param[k], Choice
            ), "Only support discrete parameterization now"
        self.param = param
        self.obj_func = obj_func_scaler(obj_func, obj_exp_offset_scale)
        self.batch_size = batch_size
        self.obj_exp_scale = obj_exp_offset_scale
        self.last_sample_internal_res = None
        self.best_sols = BestResultsQueue(MAX_NUM_BEST_SOLUTIONS)
        self._init()

    def _init(self) -> None:
        pass

    def optimize_step(self) -> Tuple:
        assert self.obj_func is not None, (
            "obj_func not provided. Can't call optimize_step() for optimization. "
            "You have to perform manual optimization, i.e., call sample_internal() then update_params()"
        )

        all_results = self._optimize_step()
        sampled_solutions, sampled_reward = all_results[0], all_results[1]
        self._maintain_best_solutions(sampled_solutions, sampled_reward)
        return all_results

    def _maintain_best_solutions(
        self, sampled_sols: Dict[str, torch.Tensor], sampled_reward: torch.Tensor
    ) -> None:
        for idx in range(len(sampled_reward)):
            r = sampled_reward[idx].item()
            sol = {k: sampled_sols[k][idx] for k in sampled_sols}
            self.best_sols.insert(r, sol)

    def best_solutions(
        self, k: int = 1
    ) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        k solutions with the smallest rewards
        Return is a list of tuples (reward, solution)
        """
        return self.best_sols.topk(k)

    @abc.abstractmethod
    def _optimize_step(self) -> Tuple:
        """
        The main component of ComboOptimizer.optimize_step(). The user only
        needs to loop over optimizer_step() until the budget runs out.

        _optimize_step() will call sample_internal() and update_params()
        to perform sampling and parameter updating
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_internal(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple:
        """
        Record and return sampled solutions and any other important
        information during learning / training. The return type is a tuple,
        whose first element is always the sampled solutions (Dict[str, torch.Tensor]).

        It samples self.batch_size number of solutions (i.e., the batch size used during
        training), unless batch_size is provided.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_params(
        self,
        reward: torch.Tensor,
    ) -> None:
        """
        Update model parameters by reward. Reward is objective function
        values evaluated on the solutions sampled by sample_internal()
        """
        raise NotImplementedError()

    def sample(
        self, batch_size: int, temp: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return sampled solutions, keyed by parameter names.
        For discrete parameters, the values are choice indices;
        For continuous parameters, the values are sampled float vectors.

        This function is usually called after learning is done.
        """
        raise NotImplementedError()

    def indices_to_raw_choices(
        self, sampled_sol: Dict[str, torch.Tensor]
    ) -> List[Dict[str, str]]:
        if not sampled_sol:
            # empty sampled_sol
            return [{} for _ in range(self.batch_size)]

        batch_size = list(sampled_sol.values())[0].shape[0]
        sampled_sol_i_vals = []
        for i in range(batch_size):
            sampled_sol_i = {k: sampled_sol[k][i] for k in sampled_sol}
            sampled_sol_i_val = {
                # pyre-fixme[16]: `Parameter` has no attribute `choices`.
                k: self.param[k].choices.value[v]
                for k, v in sampled_sol_i.items()
            }
            sampled_sol_i_vals.append(sampled_sol_i_val)
        return sampled_sol_i_vals


class RandomSearchOptimizer(ComboOptimizerBase):
    """
    Find the best solution to minimize a black-box function by random search

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

        sampling_weights (Optional[Dict[str, np.ndarray]]):
            Instead of uniform sampling, we sample solutions with preferred
            weights. Key: choice name, value: sampling weights

    Example:
        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 4
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     reward = torch.ones(BATCH_SIZE, 1)
        ...     for i in range(BATCH_SIZE):
        ...         # the best action is "red"
        ...         if sampled_sol['choice1'][i] == 2:
        ...             reward[i, 0] = 0.0
        ...     return reward
        ...
        >>> optimizer = RandomSearchOptimizer(ng_param, obj_func, batch_size=BATCH_SIZE)
        >>> for i in range(10):
        ...     res = optimizer.optimize_step()
        ...
        >>> best_reward, best_choice = optimizer.best_solutions(k=1)[0]
        >>> assert best_reward == 0
        >>> assert best_choice['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        batch_size: int = BATCH_SIZE,
        sampling_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.sampling_weights = sampling_weights
        super().__init__(
            param,
            obj_func,
            batch_size=batch_size,
        )

    def sample(
        self, batch_size: int, temp: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        assert temp is None, "temp is not used in Random Search"
        sampled_sol = {}
        for k, param in self.param.items():
            # pyre-fixme[16]: `Parameter` has no attribute `choices`.
            num_choices = len(param.choices)
            if self.sampling_weights is None:
                sampled_sol[k] = torch.randint(num_choices, (batch_size,))
            else:
                weight = self.sampling_weights[k]
                sampled_sol[k] = torch.tensor(
                    np.random.choice(num_choices, batch_size, replace=True, p=weight)
                )
        return sampled_sol

    def sample_internal(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor]]:
        batch_size = batch_size or self.batch_size
        sampled_sol = self.sample(batch_size, temp=None)
        self.last_sample_internal_res = sampled_sol
        return (sampled_sol,)

    def update_params(self, reward: torch.Tensor):
        self.last_sample_internal_res = None

    def _optimize_step(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sampled_solutions = self.sample_internal(self.batch_size)[0]
        sampled_reward, _ = self.obj_func(sampled_solutions)
        sampled_reward = sampled_reward.detach()
        self.update_params(sampled_reward)
        return sampled_solutions, sampled_reward


class NeverGradOptimizer(ComboOptimizerBase):
    """
    Minimize a black-box function using NeverGrad, Rapin & Teytaud, 2018.
    https://facebookresearch.github.io/nevergrad/.

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        estimated_budgets (int): estimated number of budgets (objective evaluation
            times) for nevergrad to perform auto tuning.

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

        optimizer_name (Optional[str]): ng optimizer to be used specifically
            All possible nevergrad optimizers are available at:
            https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer.
            If not specified, we use the meta optimizer NGOpt

    Example:

        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 4
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     reward = torch.ones(BATCH_SIZE, 1)
        ...     for i in range(BATCH_SIZE):
        ...         # the best action is "red"
        ...         if sampled_sol['choice1'][i] == 2:
        ...             reward[i, 0] = 0.0
        ...     return reward
        ...
        >>> estimated_budgets = 40
        >>> optimizer = NeverGradOptimizer(
        ...    ng_param, estimated_budgets, obj_func, batch_size=BATCH_SIZE,
        ... )
        >>>
        >>> for i in range(10):
        ...     res = optimizer.optimize_step()
        ...
        >>> best_reward, best_choice = optimizer.best_solutions(k=1)[0]
        >>> assert best_reward == 0
        >>> assert best_choice['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        estimated_budgets: int,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        batch_size: int = BATCH_SIZE,
        optimizer_name: Optional[str] = None,
    ) -> None:
        self.estimated_budgets = estimated_budgets
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.choice_to_index = {}
        super().__init__(
            param,
            obj_func,
            batch_size=batch_size,
        )

    def _init(self) -> None:
        optimizer_name = self.optimizer_name or "NGOpt"
        logger.info(f"Nevergrad uses {optimizer_name} optimizer")
        self.optimizer = ng.optimizers.registry[optimizer_name](
            parametrization=self.param,
            budget=self.estimated_budgets,
            num_workers=self.batch_size,
        )
        for k, param in self.param.items():
            # pyre-fixme[16]: `Parameter` has no attribute `choices`.
            self.choice_to_index[k] = {v: i for i, v in enumerate(param.choices.value)}

    def sample(
        self, batch_size: int, temp: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        assert temp is None, "temp is not used in Random Search"
        ng_sols_idx = {k: torch.zeros(batch_size) for k in self.param}
        for i in range(batch_size):
            ng_sol = self.optimizer.ask().value
            for k in ng_sol:
                ng_sols_idx[k][i] = self.choice_to_index[k][ng_sol[k]]
        return ng_sols_idx

    def sample_internal(self, batch_size: Optional[int] = None) -> Tuple:
        """
        Return sampled solutions in two formats.
        (1) our own format, which is a dictionary and consistent with other optimizers.
            The dictionary has choice names as the key and sampled choice indices as the
            value (of shape (batch_size, ))
        (2) nevergrad format returned by optimizer.ask()
        """
        batch_size = batch_size or self.batch_size
        ng_sols_idx = {k: torch.zeros(batch_size, dtype=torch.long) for k in self.param}
        ng_sols_raw = []
        for i in range(batch_size):
            ng_sol = self.optimizer.ask()
            ng_sols_raw.append(ng_sol)
            ng_sol_val = ng_sol.value
            for k in ng_sol_val:
                ng_sols_idx[k][i] = self.choice_to_index[k][ng_sol_val[k]]
        self.last_sample_internal_res = (ng_sols_idx, ng_sols_raw)
        return ng_sols_idx, ng_sols_raw

    def update_params(self, reward: torch.Tensor) -> None:
        _, sampled_sols = self.last_sample_internal_res
        for ng_sol, r in zip(sampled_sols, reward):
            self.optimizer.tell(ng_sol, r.item())
        self.last_sample_internal_res = None

    def _optimize_step(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sampled_sol_idxs, sampled_sols = self.sample_internal(self.batch_size)
        sampled_reward, _ = self.obj_func(sampled_sol_idxs)
        sampled_reward = sampled_reward.detach()
        self.update_params(sampled_reward)
        return sampled_sol_idxs, sampled_reward


class LogitBasedComboOptimizerBase(ComboOptimizerBase):
    def __init__(
        self,
        param: ng.p.Dict,
        start_temp: float,
        min_temp: float,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        learning_rate: float = LEARNING_RATE,
        anneal_rate: float = ANNEAL_RATE,
        batch_size: int = BATCH_SIZE,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.temp = start_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.learning_rate = learning_rate
        self.logits: Dict[str, nn.Parameter] = {}
        self.optimizer = None
        super().__init__(
            param,
            obj_func,
            batch_size,
            obj_exp_offset_scale,
        )

    def _init(self) -> None:
        parameters = []
        for k in self.param.keys():
            v = self.param[k]
            if isinstance(v, ng.p.Choice):
                logits_shape = len(v.choices)
                self.logits[k] = nn.Parameter(torch.randn(1, logits_shape))
                parameters.append(self.logits[k])
            else:
                raise NotImplementedError()
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

    def sample(
        self, batch_size: int, temp: Optional[float] = GREEDY_TEMP
    ) -> Dict[str, torch.Tensor]:
        assert temp is not None, "temp is needed for sampling logits"
        sampled_solutions, _ = sample_from_logits(self.logits, batch_size, temp)
        return sampled_solutions


def sample_gumbel(shape: Tuple[int, ...], eps: float = 1e-20) -> torch.Tensor:
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


class GumbelSoftmaxOptimizer(LogitBasedComboOptimizerBase):
    """
    Minimize a differentiable objective function which takes in categorical inputs.
    The method is based on Categorical Reparameterization with Gumbel-Softmax,
    Jang, Gu, & Poole, 2016. https://arxiv.org/abs/1611.01144.

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            an analytical function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled gumbel-softmax
            distributions of shape (batch_size, num_choices) as the value

        start_temp: starting temperature

        min_temp: minimal temperature (towards the end of learning) for sampling gumbel-softmax

        update_params_within_optimizer (bool): If False, skip updating parameters within this
            Optimizer. The Gumbel-softmax parameters will be updated in external systems.


    Example:

        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 4
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     # best action is "red"
        ...     reward = torch.mm(sampled_sol['choice1'], torch.tensor([[1.], [1.], [0.]]))
        ...     return reward
        ...
        >>> optimizer = GumbelSoftmaxOptimizer(
        ...     ng_param, obj_func, anneal_rate=0.9, batch_size=BATCH_SIZE, learning_rate=0.1
        ... )
        ...
        >>> for i in range(30):
        ...     res = optimizer.optimize_step()
        ...
        >>> assert optimizer.sample(1)['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        start_temp: float = 1.0,
        min_temp: float = 0.1,
        learning_rate: float = LEARNING_RATE,
        anneal_rate: float = ANNEAL_RATE,
        batch_size: int = BATCH_SIZE,
        update_params_within_optimizer: bool = True,
    ) -> None:
        self.update_params_within_optimizer = update_params_within_optimizer
        super().__init__(
            param,
            start_temp,
            min_temp,
            obj_func,
            learning_rate=learning_rate,
            anneal_rate=anneal_rate,
            batch_size=batch_size,
            # no reward scaling in gumbel softmax
            obj_exp_offset_scale=None,
        )

    def sample_internal(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor]]:
        batch_size = batch_size or self.batch_size
        sampled_softmax_vals = {}
        for k, logits in self.logits.items():
            sampled_softmax_vals[k] = gumbel_softmax(
                logits.repeat(batch_size, 1), self.temp
            )
        self.last_sample_internal_res = sampled_softmax_vals
        return (sampled_softmax_vals,)

    def update_params(self, reward: torch.Tensor) -> None:
        if self.update_params_within_optimizer:
            reward_mean = reward.mean()
            assert reward_mean.requires_grad
            self.optimizer.zero_grad()
            reward_mean.backward()
            self.optimizer.step()

        self.temp = np.maximum(self.temp * self.anneal_rate, self.min_temp)
        self.last_sample_internal_res = None

    def _optimize_step(self) -> Tuple:
        sampled_softmax_vals = self.sample_internal(self.batch_size)[0]
        sampled_reward, _ = self.obj_func(sampled_softmax_vals)
        self.update_params(sampled_reward)

        sampled_softmax_vals = {
            k: v.detach().clone() for k, v in sampled_softmax_vals.items()
        }
        logits = {k: v.detach().clone() for k, v in self.logits.items()}
        return sampled_softmax_vals, sampled_reward, logits


class PolicyGradientOptimizer(LogitBasedComboOptimizerBase):
    """
    Minimize a black-box objective function which takes in categorical inputs.
    The method is based on REINFORCE, Williams, 1992.
    https://link.springer.com/article/10.1007/BF00992696

    In this method, the action distribution is a joint distribution of multiple
    *independent* softmax distributions, each corresponding to one discrete
    choice type.

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

    Example:
        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 16
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     reward = torch.ones(BATCH_SIZE, 1)
        ...     for i in range(BATCH_SIZE):
        ...         # the best action is "red"
        ...         if sampled_sol['choice1'][i] == 2:
        ...             reward[i, 0] = 0.0
        ...     return reward
        ...
        >>> optimizer = PolicyGradientOptimizer(
        ...     ng_param, obj_func, batch_size=BATCH_SIZE, learning_rate=0.1
        ... )
        >>> for i in range(30):
        ...    res = optimizer.optimize_step()
        ...
        >>> best_reward, best_choice = optimizer.best_solutions(k=1)[0]
        >>> assert best_reward == 0
        >>> assert best_choice['choice1'] == 2
        >>> assert optimizer.sample(1)['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        # default (start_temp=min_temp=1.0): no temperature change for policy gradient
        start_temp: float = 1.0,
        min_temp: float = 1.0,
        learning_rate: float = LEARNING_RATE,
        anneal_rate: float = ANNEAL_RATE,
        batch_size: int = BATCH_SIZE,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            param,
            start_temp,
            min_temp,
            obj_func,
            learning_rate=learning_rate,
            anneal_rate=anneal_rate,
            batch_size=batch_size,
            obj_exp_offset_scale=obj_exp_offset_scale,
        )

    def sample(
        self, batch_size: int, temp: Optional[float] = GREEDY_TEMP
    ) -> Dict[str, torch.Tensor]:
        assert temp is not None, "temp is needed for sampling logits"
        sampled_solutions, _ = sample_from_logits(self.logits, batch_size, temp)
        return sampled_solutions

    def sample_internal(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = batch_size or self.batch_size
        sampled_solutions, sampled_log_probs = sample_from_logits(
            self.logits,
            batch_size,
            self.temp,
        )
        self.last_sample_internal_res = sampled_solutions, sampled_log_probs
        return sampled_solutions, sampled_log_probs

    def update_params(self, reward: torch.Tensor):
        _, sampled_log_probs = self.last_sample_internal_res
        if self.batch_size == 1:
            adv = reward
        else:
            adv = reward - torch.mean(reward)

        assert not adv.requires_grad
        assert sampled_log_probs.requires_grad
        assert sampled_log_probs.shape == adv.shape == reward.shape
        assert adv.ndim == 2
        assert adv.shape[-1] == 1

        loss = (adv * sampled_log_probs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.temp = np.maximum(self.temp * self.anneal_rate, self.min_temp)
        self.last_sample_internal_res = None

    def _optimize_step(self) -> Tuple:
        sampled_solutions, sampled_log_probs = self.sample_internal(self.batch_size)

        sampled_reward, sampled_scaled_reward = self.obj_func(sampled_solutions)
        sampled_reward, sampled_scaled_reward = (
            sampled_reward.detach(),
            sampled_scaled_reward.detach(),
        )
        self.update_params(sampled_scaled_reward)
        return sampled_solutions, sampled_reward, sampled_log_probs


def shuffle_exp_replay(exp_replay: List[Any]) -> Any:
    shuffle_idx = np.random.permutation(len(exp_replay))
    for idx in shuffle_idx:
        yield exp_replay[idx]


class QLearningOptimizer(ComboOptimizerBase):
    """
    Treat the problem of minimizing a black-box function as a sequential decision problem,
    and solve it by Deep Q-Learning. See "Human-Level Control through Deep Reinforcement
    Learning", Mnih et al., 2015. https://www.nature.com/articles/nature14236.

    In each episode step, Q-learning makes a decision for one categorical input. The reward
    is given only at the end of the episode, which is the value of the black-box function
    at the input determined by the choices made at all steps.

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        start_temp (float): the starting exploration rate in epsilon-greedy sampling

        min_temp (float): the minimal exploration rate in epsilon-greedy

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

        model_dim (int): hidden layer size for the q-network: input -> model_dim -> model_dim -> output

        num_batches_per_learning (int): the number of batches sampled from replay buffer
            for q-learning.

        replay_size (int): the maximum batches held in the replay buffer. Note, a problem instance of n
            choices will generate n batches in the replay buffer.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 4
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     reward = torch.ones(BATCH_SIZE, 1)
        ...     for i in range(BATCH_SIZE):
        ...         # the best action is "red"
        ...         if sampled_sol['choice1'][i] == 2:
        ...             reward[i, 0] = 0.0
        ...     return reward
        ...
        >>> optimizer = QLearningOptimizer(ng_param, obj_func, batch_size=BATCH_SIZE)
        >>> for i in range(10):
        ...     res = optimizer.optimize_step()
        ...
        >>> best_reward, best_choice = optimizer.best_solutions(k=1)[0]
        >>> assert best_reward == 0
        >>> assert best_choice['choice1'] == 2
        >>> assert optimizer.sample(1)['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        start_temp: float = 1.0,
        min_temp: float = 0.1,
        learning_rate: float = LEARNING_RATE,
        anneal_rate: float = ANNEAL_RATE,
        batch_size: int = BATCH_SIZE,
        model_dim: int = 128,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
        num_batches_per_learning: int = 10,
        replay_size: int = 100,
    ) -> None:
        self.model_dim = model_dim
        self.sorted_keys = sorted(param.keys())
        assert (
            start_temp <= 1.0 and start_temp > 0
        ), "Starting temperature for epsilon-greedy should be between (0, 1]"
        assert (
            min_temp <= start_temp and min_temp >= 0
        ), "Minimum temperature for epsilon-greedy should be between [0, start_temp]"
        self.temp = start_temp
        self.min_temp = min_temp
        self.learning_rate = learning_rate
        self.anneal_rate = anneal_rate
        self.num_batches_per_learning = num_batches_per_learning
        self.replay_size = replay_size
        self.exp_replay = deque([], maxlen=replay_size)
        self.input_dim = 0
        self.q_net = None
        self.optimizer = None
        super().__init__(
            param,
            obj_func,
            batch_size=batch_size,
            obj_exp_offset_scale=obj_exp_offset_scale,
        )

    def _init(self) -> None:
        for k in self.sorted_keys:
            v = self.param[k]
            if isinstance(v, ng.p.Choice):
                num_choices = len(v.choices)
                self.input_dim += num_choices
            else:
                raise NotImplementedError()

        self.q_net = nn.Sequential(
            *[
                nn.Linear(self.input_dim, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, 1),
            ]
        )
        for p in self.q_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )

        logger.info(f"Number of total params: {_num_of_params(self.q_net)}")

    def sample_internal(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
        batch_size = batch_size or self.batch_size
        return self._sample_internal(batch_size, self.temp)

    def _sample_internal(
        self,
        batch_size: int,
        temp: float,
    ) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
        logger.info(f"Explore with temp={temp}")
        sampled_solutions: Dict[str, torch.Tensor] = {}
        exp_replay = []
        acc_input_dim = 0
        # The first cur_state_action is a dummy vector of all -1
        cur_state_action = torch.full((batch_size, self.input_dim), -1).float()
        for k in self.sorted_keys:
            v = self.param[k]
            # pyre-fixme[16]: `Parameter` has no attribute `choices`.
            num_choices = len(v.choices)
            next_state_action_all_pairs = cur_state_action.repeat_interleave(
                num_choices, dim=0
            ).reshape(batch_size, num_choices, self.input_dim)
            next_state_action_all_pairs[
                :, :, acc_input_dim : acc_input_dim + num_choices
            ] = torch.eye(num_choices)
            q_values = (
                self.q_net(next_state_action_all_pairs)
                .detach()
                .reshape(batch_size, num_choices)
            )
            q_actions = q_values.argmax(dim=1)
            random_actions = torch.randint(num_choices, (batch_size,))
            explore_prob = torch.rand(batch_size)
            selected_action = (
                (explore_prob <= temp) * random_actions
                + (explore_prob > temp) * q_actions
            ).long()

            sampled_solutions[k] = selected_action
            # the last element is terminal indicator
            exp_replay.append((cur_state_action, next_state_action_all_pairs, False))

            cur_state_action = next_state_action_all_pairs[
                torch.arange(batch_size), selected_action
            ]
            acc_input_dim += num_choices

        # add dummy next_state_action_all_pairs and terminal indicator
        exp_replay.append((cur_state_action, cur_state_action.squeeze(1), True))
        # the first element is not useful
        exp_replay.pop(0)

        self.last_sample_internal_res = (sampled_solutions, exp_replay)
        return sampled_solutions, exp_replay

    def sample(
        self, batch_size: int, temp: Optional[float] = GREEDY_TEMP
    ) -> Dict[str, torch.Tensor]:
        assert temp is not None, "temp is needed for epsilon greedy"
        sampled_solutions, _ = self._sample_internal(batch_size, temp)
        return sampled_solutions

    def update_params(self, reward: torch.Tensor) -> None:
        _, exp_replay = self.last_sample_internal_res

        # insert reward placeholder to exp replay
        # exp replay now has the format:
        # (cur_state_action, next_state_action_all_pairs, terminal, reward)
        self.exp_replay.extend([[*exp, None] for exp in exp_replay])
        self.exp_replay[-1][-1] = reward

        assert len(exp_replay) == len(self.sorted_keys)
        avg_td_loss = []

        for i, (
            cur_state_action,
            next_state_action_all_pairs,
            terminal,
            r,
        ) in enumerate(shuffle_exp_replay(self.exp_replay)):
            q = self.q_net(cur_state_action)
            if terminal:
                # negate reward to be consistent with other optimizers.
                # reward returned by obj_func is to be minimized
                # but q-learning tries to maxmize accumulated rewards
                loss = F.mse_loss(q, -r)
            else:
                q_next = self.q_net(next_state_action_all_pairs).detach()
                # assume gamma=1 (no discounting)
                loss = F.mse_loss(q, q_next.max(dim=1).values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_td_loss.append(loss.detach())

            if i == self.num_batches_per_learning - 1:
                break

        avg_td_loss = np.mean(avg_td_loss)
        logger.info(f"Avg td loss: {avg_td_loss}")

        self.temp = np.maximum(self.temp * self.anneal_rate, self.min_temp)
        self.last_sample_internal_res = None

    def _optimize_step(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sampled_solutions, exp_replay = self.sample_internal(self.batch_size)
        sampled_reward, sampled_scaled_reward = self.obj_func(sampled_solutions)
        sampled_reward, sampled_scaled_reward = (
            sampled_reward.detach(),
            sampled_scaled_reward.detach(),
        )
        self.update_params(sampled_scaled_reward)
        return sampled_solutions, sampled_reward


class BayesianOptimizerBase(ComboOptimizerBase):
    """
    Bayessian Optimization with mutation optimization and acquisition function.
    The method is motivated from BANANAS, White, 2020.
    https://arxiv.org/abs/1910.11858

    In this method, the searching is based on mutation over the current best solutions.
    Acquisition function, e.g., its estimates the expected imrpovement.

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

        acq_type (str): type of acquisition function.

        mutation_type (str): type of mutation, e.g., random.

        temp (float): percentage of mutation - how many variables will be mutated.
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        start_temp: float = 1.0,
        min_temp: float = 0.1,
        acq_type: str = "its",
        mutation_type: str = "random",
        anneal_rate: float = ANNEAL_RATE,
        batch_size: int = BATCH_SIZE,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.temp = start_temp
        self.acq_type = acq_type
        self.mutation_type = mutation_type
        self.anneal_rate = anneal_rate
        super().__init__(
            param,
            obj_func,
            batch_size=batch_size,
            obj_exp_offset_scale=obj_exp_offset_scale,
        )

    def sample(
        self, batch_size: int, temp: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Applies a type of mutation, e.g., random mutation, on the best solutions recorded so far.
        For example, with random mutation, variables are randomly selected,
        and their values are randomly set with respect to their domains.
        """
        assert temp is not None, "temperature is needed for Bayesian Optimizer"
        best_solutions = self.best_solutions(batch_size)
        # best_solutions come in as (reward, solution) tuples
        # we only need solutions so we strip reward
        best_solutions = [sol for _, sol in best_solutions]
        if len(best_solutions) < batch_size:
            logger.warning(
                "Less than batch_size solutions are sampled to be mutated. Will duplicate thse solutions."
            )
            dup_times = batch_size // len(best_solutions) + 1
            best_solutions = (best_solutions * dup_times)[:batch_size]
            assert batch_size == len(best_solutions)

        # Convert best_solutions to Dict[str, tensor] format
        sampled_solutions = {}
        for k in sorted(self.param.keys()):
            sampled_solutions[k] = torch.cat(
                [sol[k].reshape(1) for sol in best_solutions]
            )

        if self.mutation_type == "random":
            # keys to mutate for each solution
            mutated_keys = [
                np.random.choice(
                    sorted(self.param.keys()),
                    floor(temp * len(self.param)),
                    replace=False,
                )
                for _ in range(batch_size)
            ]
            mutated_solutions = {}
            for key in sorted(self.param.keys()):
                mutated_solutions[key] = sampled_solutions[key].clone()
                sol_indices = torch.tensor(
                    [
                        sol_idx
                        for sol_idx, mutated_keys_for_one_sol in enumerate(mutated_keys)
                        if key in mutated_keys_for_one_sol
                    ]
                )
                if len(sol_indices):
                    mutated_solutions[key][sol_indices] = torch.randint(
                        # pyre-fixme[16]: `Parameter` has no attribute `choices`.
                        len(self.param[key].choices),
                        (len(sol_indices),),
                    )
        else:
            raise NotImplementedError()
        return mutated_solutions

    def acquisition(
        self,
        acq_type: str,
        sampled_sol: Dict[str, torch.Tensor],
        predictor: List[nn.Module],
    ) -> torch.Tensor:
        assert predictor is not None
        batch_tensors = sol_to_tensors(sampled_sol, self.param)
        if acq_type == "its":
            with torch.no_grad():
                predictions = torch.stack([net(batch_tensors) for net in predictor])
                acquisition_reward = torch.normal(
                    torch.mean(predictions, dim=0), torch.std(predictions, dim=0)
                )
        else:
            raise NotImplementedError()
        return acquisition_reward.view(-1)


class BayesianMLPEnsemblerOptimizer(BayesianOptimizerBase):
    """
    Bayessian Optimizer with ensemble of mlp networks, random mutation, and ITS.
    The Method is motivated by the BANANAS optimization method, White, 2019.
    https://arxiv.org/abs/1910.11858.

    The mutation rate (temp) is starting from start_temp and is decreasing over time
    with anneal_rate. It's lowest possible value is min_temp.
    Thus, initially the algorithm explores mutations with a higer mutation rate (more variables are randomly mutated).
    As time passes, the algorithm exploits the best solutions recorded so far (less variables are mutated).

    Args:
        param (ng.p.Dict): a nevergrad dictionary for specifying input choices

        obj_func (Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
            a function which consumes sampled solutions and returns
            rewards as tensors of shape (batch_size, 1).

            The input dictionary has choice names as the key and sampled choice
            indices as the value (of shape (batch_size, ))

        acq_type (str): type of acquisition function.

        mutation_type (str): type of mutation, e.g., random.

        num_mutations (int): number of best solutions recorded so far that will be mutated.

        num_ensemble (int): number of predictors.

        start_temp (float): initial temperature (ratio) for mutation, e.g., with 1.0 all variables will be initally mutated.

        min_temp (float): lowest temperature (ratio) for mutation, e.g., with 0.0 no mutation will occur.


    Example:
        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> BATCH_SIZE = 4
        >>> ng_param = ng.p.Dict(choice1=ng.p.Choice(["blue", "green", "red"]))
        >>>
        >>> def obj_func(sampled_sol: Dict[str, torch.Tensor]):
        ...     reward = torch.ones(BATCH_SIZE, 1)
        ...     for i in range(BATCH_SIZE):
        ...         # the best action is "red"
        ...         if sampled_sol['choice1'][i] == 2:
        ...             reward[i, 0] = 0.0
        ...     return reward
        ...
        >>> optimizer = BayesianMLPEnsemblerOptimizer(
        ...     ng_param, obj_func, batch_size=BATCH_SIZE,
        ...     acq_type="its", mutation_type="random",
        ...     num_mutations=4,
        ... )
        >>> for i in range(30):
        ...     res = optimizer.optimize_step()
        ...
        >>> assert optimizer.sample(1, temp=0)['choice1'] == 2
    """

    def __init__(
        self,
        param: ng.p.Dict,
        obj_func: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        start_temp: float = 1.0,
        min_temp: float = 0.1,
        acq_type: str = "its",
        mutation_type: str = "random",
        anneal_rate: float = ANNEAL_RATE,
        num_mutations: int = 50,
        epochs: int = 1,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        obj_exp_offset_scale: Optional[Tuple[float, float]] = None,
        model_dim: int = 128,
        num_ensemble: int = 5,
    ) -> None:
        self.temp = start_temp
        self.num_mutations = num_mutations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_dim = model_dim
        self.num_ensemble = num_ensemble
        self.input_dim = 0
        self.predictor = None
        self.last_predictor_loss_mean = None
        super().__init__(
            param,
            obj_func,
            start_temp=start_temp,
            min_temp=min_temp,
            acq_type=acq_type,
            mutation_type=mutation_type,
            anneal_rate=anneal_rate,
            batch_size=batch_size,
            obj_exp_offset_scale=obj_exp_offset_scale,
        )

    def _init(self) -> None:
        # initial population
        sampled_solutions = {}
        for k, param in self.param.items():
            if isinstance(param, ng.p.Choice):
                num_choices = len(param.choices)
                self.input_dim += num_choices
                sampled_solutions[k] = torch.randint(num_choices, (self.num_mutations,))
            else:
                raise NotImplementedError()
        # predictor
        self.predictor = []
        for _ in range(self.num_ensemble):
            model = nn.Sequential(
                *[
                    nn.Linear(self.input_dim, self.model_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.model_dim, self.model_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.model_dim, 1),
                ]
            )
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.predictor.append(model)

        sampled_reward, _ = self.obj_func(sampled_solutions)
        sampled_reward = sampled_reward.detach()
        self._maintain_best_solutions(sampled_solutions, sampled_reward)
        self.update_predictor(sampled_solutions, sampled_reward)

    def sample_internal(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        batch_size = batch_size or self.batch_size
        mutated_solutions = self.sample(self.num_mutations, self.temp)
        _, indices = torch.sort(
            self.acquisition(self.acq_type, mutated_solutions, self.predictor), dim=0
        )
        sampled_solutions = {}
        for key in sorted(self.param.keys()):
            sampled_solutions[key] = mutated_solutions[key][indices[:batch_size]]
        self.last_sample_internal_res = sampled_solutions
        return (sampled_solutions,)

    def update_predictor(
        self, sampled_solutions: Dict[str, torch.Tensor], sampled_reward: torch.Tensor
    ):
        x = sol_to_tensors(sampled_solutions, self.param)
        y = sampled_reward
        losses = []
        for model in self.predictor:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for _ in range(self.epochs):
                pred = model(x)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.detach())
            model.eval()
        self.last_predictor_loss_mean = np.mean(losses)

    def update_params(self, reward: torch.Tensor):
        sampled_solutions = self.last_sample_internal_res
        self.update_predictor(sampled_solutions, reward)
        self.temp = np.maximum(self.temp * self.anneal_rate, self.min_temp)
        self.last_sample_internal_res = None

    def _optimize_step(self) -> Tuple:
        sampled_solutions = self.sample_internal(self.batch_size)[0]
        sampled_reward, _ = self.obj_func(sampled_solutions)
        sampled_reward = sampled_reward.detach()
        self.update_params(sampled_reward)

        last_predictor_loss_mean = self.last_predictor_loss_mean
        self.last_predictor_loss_mean = None

        return sampled_solutions, sampled_reward, last_predictor_loss_mean
