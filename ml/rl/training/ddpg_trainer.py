#!/usr/bin/env python3


from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters
from ml.rl.training.ddpg_predictor import (
    DDPGPredictor
)


class DDPGTrainer():
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        num_features: int,
        action_dim: int,
    ) -> None:

        self.reward_burnin = parameters.rl.reward_burnin
        self.maxq_learning = parameters.rl.maxq_learning
        self.rl_discount_rate = parameters.rl.gamma
        self.rl_temperature = parameters.rl.temperature
        self.training_iteration = 0
        self.minibatch_size = parameters.training.minibatch_size
        self.parameters = parameters
        self.parameters.training.layers[0] = num_features
        self.parameters.training.layers[-1] = action_dim

    def train(self):
        pass

    def predictor(self) -> DDPGPredictor:
        """
        Builds a ContinuousActionPredictor using the MLTrainer underlying this
        ContinuousActionTrainer.
        """
        return DDPGPredictor.export(self)
