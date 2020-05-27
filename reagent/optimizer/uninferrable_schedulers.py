#!/usr/bin/env python3

"""
This file contains configs that could not be inferred from the default values
provided by PyTorch. If PyTorch optimizers and lr_schedulers had type annotations
then we could infer everything.
default values that cannot be inferred:
- tuple
- None
- required parameters (no default value)
TODO: remove this file once we can infer everything.
"""
from typing import List, Optional, Union

from reagent.core.dataclasses import dataclass
from reagent.parameters import param_hash

from .scheduler import LearningRateSchedulerConfig


@dataclass(frozen=True)
class CyclicLR(LearningRateSchedulerConfig):
    # scale_fn is Callable, which FBL doesn't support.
    # TODO(T67530507) Add a scale function factory (FBL doesn't allow callables)
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
    __hash__ = param_hash

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
