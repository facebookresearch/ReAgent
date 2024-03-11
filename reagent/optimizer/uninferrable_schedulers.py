#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

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
from typing import Any, Callable, Dict, List, Optional, Union

from reagent.core.dataclasses import dataclass
from reagent.core.fb_checker import IS_FB_ENVIRONMENT

from .scheduler import LearningRateSchedulerConfig

# Inside FB, we have more sophisticated classes to serialize Callables
if not IS_FB_ENVIRONMENT:

    # To allow string-based configuration, we need these Mixins to convert
    # from strings to Callables
    class _LRLambdaMixin:
        def decode_lambdas(self, args: Dict[str, Any]) -> None:
            lr_lambda = args.get("lr_lambda")
            if type(lr_lambda) is str:
                args["lr_lambda"] = eval(lr_lambda)  # noqa

    class _ScaleFnLambdaMixin:
        def decode_lambdas(self, args: Dict[str, Any]) -> None:
            scale_fn = args.get("scale_fn")
            if type(scale_fn) is str:
                args["scale_fn"] = eval(scale_fn)  # noqa

    @dataclass(frozen=True)
    class LambdaLR(_LRLambdaMixin, LearningRateSchedulerConfig):
        lr_lambda: Union[str, Callable[[int], float], List[Callable[[int], float]]]
        last_epoch: int = -1
        verbose: bool = False

    @dataclass(frozen=True)
    class MultiplicativeLR(_LRLambdaMixin, LearningRateSchedulerConfig):
        lr_lambda: Union[str, Callable[[int], float], List[Callable[[int], float]]]
        last_epoch: int = -1
        verbose: bool = False

    @dataclass(frozen=True)
    class CyclicLR(_ScaleFnLambdaMixin, LearningRateSchedulerConfig):
        base_lr: Union[float, List[float]]
        max_lr: Union[float, List[float]]
        step_size_up: int = 2000
        step_size_down: Optional[int] = None
        mode: str = "triangular"
        gamma: float = 1.0
        scale_fn: Optional[Union[str, Callable[[int], float]]] = None
        scale_mode: str = "cycle"
        cycle_momentum: bool = True
        base_momentum: float = 0.8
        max_momentum: float = 0.9
        last_epoch: int = -1
        verbose: bool = False


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
