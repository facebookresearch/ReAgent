#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from typing import Dict

import reagent.serving.config.namespace as namespace


# Track things we don't want to export
__globals = set(globals())


def ActionValueScoring(model_id: int, snapshot_id: int):
    """The operator calls predictor for a model and generates the scores

    Args:

    """
    pass


def EpsilonGreedyRanker(epsilon: float, values: Dict, seed: int = None):
    """The operator that implements iterative epsilon greedy ranker

    Args:

    """
    pass


def Expression(equation: str):
    """The operator that can do calculation based on the given equation

    Args:

    """
    pass


def Frechet(values: Dict, rho: float, gamma: float, seed: int = None):
    """The operator that implements Frechet ranking

    Args:

    """
    pass


def InputFromRequest():
    """The operator that will return "input" from request

    Args:

    """
    pass


def PropensityFit(input: Dict, targets: Dict):
    """The operator that shifts the input to match the distribution in targets

    Args:

    """
    pass


def Softmax(temperature: float, values: Dict):
    """The operator to use softmax for normalization

    Args:

    """
    pass


def SoftmaxRanker(temperature: float, values: Dict, seed: int = None):
    """The operator that implements iterative softmax ranker

    Args:

    """
    pass


def UCB(method: str, batch_size: int = 1):
    """The operator that implements UCB algorithms

    Args:

    """
    pass


# Set exports
for op in set(globals()) - __globals - {"__globals"}:
    globals()[op] = namespace.DecisionOperation(globals()[op])
del __globals
