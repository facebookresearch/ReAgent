#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import reagent.serving.config.namespace as namespace

# Track things we don't want to export
__globals = set(globals())


def PredictorScoring(
    model_id,
    snapshot_id,
):
    """The operator calls predictor for a model and generates the scores

    Args:

    """
    pass


def Softmax(
    temperature,
    values,
):
    """The operator to use softmax for normalization

    Args:

    """
    pass


def SoftmaxRanker(
    temperature,
    values,
    seed=None,
):
    """The operator that implements iterative softmax ranker

    Args:

    """
    pass


def EpsilonGreedyRanker(
    epsilon,
    values,
    seed=None,
):
    """The operator that implements iterative epsilon greedy ranker

    Args:

    """
    pass


def Frechet(
    values,
    rho,
    gamma,
    seed=None,
):
    """The operator that implements Frechet ranking

    Args:

    """
    pass


def InputFromRequest():
    """The operator that will return "input" from request

    Args:

    """
    pass


# Set exports
for op in set(globals()) - __globals - {"__globals"}:
    globals()[op] = namespace.DecisionOperation(globals()[op])
del __globals
