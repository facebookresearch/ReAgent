#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from reagent.serving.config.builder import (
    UCB,
    DecisionPlanBuilder,
    EpsilonGreedyRanker,
    Frechet,
    InputFromRequest,
    Softmax,
    SoftmaxRanker,
    export,
)


def softmax_decision_plan():
    op = Softmax(temperature=1.0, values={"action1": 10.0, "action2": 20.0})
    return DecisionPlanBuilder().set_root(op).build()


def softmaxranker_decision_plan():
    op = SoftmaxRanker(temperature=1.0, values={"Bacon": 1.1, "Ribs": 1.0})
    return DecisionPlanBuilder().set_root(op).build()


def epsilongreedyranker_decision_plan():
    op = EpsilonGreedyRanker(epsilon=0.1, values={"action1": 10.0, "action2": 20.0})
    return DecisionPlanBuilder().set_root(op).build()


def frechet_decision_plan():
    op = Frechet(rho=0.5, gamma=1.0, values={"action1": 10.0, "action2": 20.0})
    return DecisionPlanBuilder().set_root(op).build()


def ucb_decision_plan():
    op = UCB(method="UCB1", batch_size=8)
    return DecisionPlanBuilder().set_root(op).build()


def input_from_request_decision_plan():
    op = Softmax(temperature=1.0, values=InputFromRequest())
    return DecisionPlanBuilder().set_root(op).build()


export(
    app_id="example",
    configs={
        "softmax": softmax_decision_plan(),
        "softmaxranker": softmaxranker_decision_plan(),
        "epsilongreedyranker": epsilongreedyranker_decision_plan(),
        "frechet": frechet_decision_plan(),
        "ucb": ucb_decision_plan(),
        "input_from_request": input_from_request_decision_plan(),
    },
)
