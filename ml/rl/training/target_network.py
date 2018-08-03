#!/usr/bin/env python3


import numpy as np
from caffe2.python import model_helper, workspace
from ml.rl.thrift.core.ttypes import TrainingParameters
from ml.rl.training.dnn import DNN
from ml.rl.training.ml_trainer import MLTrainer


# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# ]


class TargetNetwork(DNN):
    """ The target network is used to compute the labels in deep TD learning.
        This class computes the labels and updates the weights of the target
        network
    """

    def __init__(
        self,
        name: str,
        parameters: TrainingParameters,
        target_update_rate: float,
        source_trainer: MLTrainer,
    ) -> None:
        self._target_update_rate = target_update_rate
        self.enabled_slow_updates = False

        DNN.__init__(self, name, parameters)

        self._setup_update_net(source_trainer)

        workspace.RunNetOnce(self._update_model.param_init_net)
        workspace.CreateNet(self._update_model.net)

    def _setup_initial_blobs(self):
        self._update_rate_blob = self.model_id + "_update_rate"
        workspace.FeedBlob(self._update_rate_blob, np.array([1], dtype=np.float32))
        self._retain_rate_blob = self.model_id + "_retain_rate"
        workspace.FeedBlob(self._retain_rate_blob, np.array([0], dtype=np.float32))

        DNN._setup_initial_blobs(self)

    def _setup_update_net(self, source_trainer: MLTrainer):
        self._update_model = model_helper.ModelHelper(
            name="TargetUpdateModel_" + self.model_id
        )
        self._add_update_ops(source_trainer.weights, self.weights)
        self._add_update_ops(source_trainer.biases, self.biases)

    def _add_update_ops(self, source_params, target_params):
        for source_param, target_param in zip(source_params, target_params):
            self._update_model.net.WeightedSum(
                [
                    target_param,
                    self._retain_rate_blob,
                    source_param,
                    self._update_rate_blob,
                ],
                [target_param],
            )

    def enable_slow_updates(self):
        if not self.enabled_slow_updates:
            workspace.FeedBlob(
                self._update_rate_blob,
                np.array([self._target_update_rate], dtype=np.float32),
            )
            workspace.FeedBlob(
                self._retain_rate_blob,
                np.array([1 - self._target_update_rate], dtype=np.float32),
            )
            self.enabled_slow_updates = True
