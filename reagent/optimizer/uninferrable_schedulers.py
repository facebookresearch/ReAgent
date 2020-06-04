#!/usr/bin/env python3

"""
This file contains configs that could not be inferred from the default values
provided by PyTorch. If PyTorch optimizers and lr_schedulers had type annotations
then we could infer everything.
default values that cannot be inferred:
- tuple
- None
- required parameters (no default value)

Sometimes there are no defaults to infer from, so we got to include those here.
TODO: remove this file once we can infer everything.
"""
from typing import List, Optional, Union

from reagent.core.dataclasses import dataclass

from .scheduler import LearningRateSchedulerConfig


@dataclass(frozen=True)
class LambdaLR(LearningRateSchedulerConfig):
    # lr_lambda is Callable, FBL doesn't support
    # TODO(T67530507) Add function factory (FBL doesn't allow callables)
    pass


@dataclass(frozen=True)
class MultiplicativeLR(LearningRateSchedulerConfig):
    # lr_lambda is Callable, FBL doesn't support
    # TODO(T67530507) Add function factory (FBL doesn't allow callables)
    pass


@dataclass(frozen=True)
class StepLR(LearningRateSchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass(frozen=True)
class MultiStepLR(LearningRateSchedulerConfig):
    milestones: List[int]
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass(frozen=True)
class ExponentialLR(LearningRateSchedulerConfig):
    gamma: float
    last_epoch: int = -1


@dataclass(frozen=True)
class CosineAnnealingLR(LearningRateSchedulerConfig):
    T_max: int
    eta_min: float = 0
    last_epoch: int = -1


@dataclass(frozen=True)
class CyclicLR(LearningRateSchedulerConfig):
    # scale_fn is Callable, which FBL doesn't support.
    # TODO(T67530507) Add function factory (FBL doesn't allow callables)
    pass
    # base_lr: Union[float, List[float]]
    # max_lr: Union[float, List[float]]
    # step_size_up: int = 2000
    # step_size_down: Optional[int] = None
    # mode: str = "triangular"
    # gamma: float = 1.0
    # scale_fn: Optional[Callable[[int], float]] = None
    # scale_mode: str = "cycle"
    # cycle_momentum: bool = True
    # base_momentum: float = 0.8
    # max_momentum: float = 0.9
    # last_epoch: int = -1


@dataclass(frozen=True)
class OneCycleLR(LearningRateSchedulerConfig):
    max_lr: Union[float, List[float]]
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    last_epoch: int = -1


@dataclass(frozen=True)
class CosineAnnealingWarmRestarts(LearningRateSchedulerConfig):
    T_0: int
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1
