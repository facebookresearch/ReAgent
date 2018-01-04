#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ml.rl.training.target_network import TargetNetwork
from ml.rl.thrift.core.ttypes import CNNModelParameters,\
    DiscreteActionModelParameters
from ml.rl.training.conv.ml_conv_trainer import MLConvTrainer
from ml.rl.training.rl_trainer import RLTrainer


class RLConvTrainer(MLConvTrainer, RLTrainer):
    def __init__(
        self,
        fc_parameters: DiscreteActionModelParameters,
        cnn_parameters: CNNModelParameters,
        img_height: int,
        img_width: int,
    ) -> None:
        MLConvTrainer.__init__(
            self, "ml_conv_trainer", fc_parameters.training, cnn_parameters,
            img_height, img_width
        )

        self.target_network = TargetNetwork(
            self, fc_parameters.rl.target_update_rate
        )

        self.reward_burnin = fc_parameters.rl.reward_burnin
        self.maxq_learning = fc_parameters.rl.maxq_learning
        self.rl_discount_rate = fc_parameters.rl.gamma

        self.training_iteration = 0
        self._buffers = None
        self.minibatch_size = fc_parameters.training.minibatch_size

        self.skip_normalization = True
