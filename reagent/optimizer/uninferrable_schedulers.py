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
class StepLR(LearningRateSchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class MultiStepLR(LearningRateSchedulerConfig):
    milestones: List[int]
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class ExponentialLR(LearningRateSchedulerConfig):
    gamma: float
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class CosineAnnealingLR(LearningRateSchedulerConfig):
    T_max: int
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


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
    three_phase: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class CosineAnnealingWarmRestarts(LearningRateSchedulerConfig):
    T_0: int
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False
