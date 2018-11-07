#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import importlib
import os


if importlib.util.find_spec("thriftpy") is not None:
    # We have thriftpy, use it
    import thriftpy  # pylint: disable=all

    thrift_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "thrift/core.thrift"
    )
    thrift = thriftpy.load(
        thrift_path, module_name="core_thrift"
    )  # pylint: disable=all
else:
    # Fall back to using the compiled thrift code
    import ml.rl.thrift.core.ttypes as thrift  # pylint: disable=all

"""
# This could add all structs automatically but confuses linters
for x in dir(thrift):
  if x[0] != '_':
    globals()[x] = thrift.__dict__[x]
"""

AdditionalFeatureTypes = thrift.AdditionalFeatureTypes
RLParameters = thrift.RLParameters
RainbowDQNParameters = thrift.RainbowDQNParameters
CNNParameters = thrift.CNNParameters
FeedForwardParameters = thrift.FeedForwardParameters
FactorizationParameters = thrift.FactorizationParameters
TrainingParameters = thrift.TrainingParameters
InTrainingCPEParameters = thrift.InTrainingCPEParameters
EvolutionParameters = thrift.EvolutionParameters
ActionBudget = thrift.ActionBudget
StateFeatureParameters = thrift.StateFeatureParameters
DiscreteActionModelParameters = thrift.DiscreteActionModelParameters
ContinuousActionModelParameters = thrift.ContinuousActionModelParameters
DDPGNetworkParameters = thrift.DDPGNetworkParameters
DDPGTrainingParameters = thrift.DDPGTrainingParameters
DDPGModelParameters = thrift.DDPGModelParameters
OptimizerParameters = thrift.OptimizerParameters
SACTrainingParameters = thrift.SACTrainingParameters
SACModelParameters = thrift.SACModelParameters
KNNDQNModelParameters = thrift.KNNDQNModelParameters
OpenAIGymParameters = thrift.OpenAIGymParameters
