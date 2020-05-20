#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


# TODO: delete after OpenAIGymEnvironment is removed
class OnPolicyPredictor(object):
    """
    This class generates actions given a trainer and a state.  It's used for
    on-policy learning.  If you have a TorchScript (i.e. serialized) model,
    Use the classes in off_policy_predictor.py
    """

    def __init__(self, trainer, action_dim: int, use_gpu: bool):
        self.trainer = trainer
        self.action_dim = action_dim
        self.use_gpu = use_gpu

    def policy_net(self) -> bool:
        """
        Return True if this predictor is for a policy network
        """
        raise NotImplementedError()

    def discrete_action(self) -> bool:
        """
        Return True if this predictor is for a discrete action network
        """
        raise NotImplementedError()
