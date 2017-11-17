from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# ]

from caffe2.python import model_helper, workspace
from ml.rl.training.ml_trainer import \
    MakeForwardPassOps


class TargetNetwork(object):
    """ The target network is used to compute the labels in deep TD learning.
        This class computes the labels and updates the weights of the target
        network
    """

    def __init__(self, trainer, target_update_rate):
        self._target_update_rate = target_update_rate
        self._update_rate_blob = trainer.model_id + "_target_update_rate"
        workspace.FeedBlob(
            self._update_rate_blob, np.array([1], dtype=np.float32)
        )
        self._retain_rate_blob = trainer.model_id + "_target_retain_rate"
        workspace.FeedBlob(
            self._retain_rate_blob, np.array([0], dtype=np.float32)
        )

        self._trainer = trainer
        self._weights = [weight + "_target" for weight in trainer.weights]
        for target_weight, source_weight in zip(self._weights, trainer.weights):
            workspace.FeedBlob(
                target_weight, workspace.FetchBlob(source_weight)
            )
        self._biases = [bias + "_target" for bias in trainer.biases]
        for target_bias, source_bias in zip(self._biases, trainer.biases):
            workspace.FeedBlob(target_bias, workspace.FetchBlob(source_bias))

        self._update_model = model_helper.ModelHelper(
            name="TargetUpdateModel_" + trainer.model_id
        )
        self._predict_model = model_helper.ModelHelper(
            name="TargetPredictModel_" + trainer.model_id
        )

        self.input_blob = "TargetInput_" + trainer.model_id
        workspace.FeedBlob(self.input_blob, np.zeros(1, dtype=np.float32))
        self.output_blob = "TargetOutput_" + trainer.model_id
        workspace.FeedBlob(self.output_blob, np.zeros(1, dtype=np.float32))

        MakeForwardPassOps(
            self._predict_model, trainer.model_id + "_target", self.input_blob,
            self.output_blob, self._weights, self._biases, trainer.activations,
            trainer.layers, trainer.dropout_ratio, True
        )
        workspace.RunNetOnce(self._predict_model.param_init_net)
        workspace.CreateNet(self._predict_model.net)

        for source_weight, target_weight in zip(trainer.weights, self._weights):
            self._update_model.net.WeightedSum(
                [
                    target_weight, self._retain_rate_blob, source_weight,
                    self._update_rate_blob
                ], [target_weight]
            )
        for source_bias, target_bias in zip(trainer.biases, self._biases):
            self._update_model.net.WeightedSum(
                [
                    target_bias, self._retain_rate_blob, source_bias,
                    self._update_rate_blob
                ], [target_bias]
            )
        workspace.RunNetOnce(self._update_model.param_init_net)
        workspace.CreateNet(self._update_model.net)

    def enable_slow_updates(self):
        workspace.FeedBlob(
            self._update_rate_blob,
            np.array([self._target_update_rate], dtype=np.float32)
        )
        workspace.FeedBlob(
            self._retain_rate_blob,
            np.array([1 - self._target_update_rate], dtype=np.float32)
        )

    def target_update(self):
        """ Updates the weights of the target network according to the
            target_update_rate
        """
        workspace.RunNetOnce(self._update_model.net)

    def target_values(self, states):
        """ Estimates the values for the given states using the target network

        :param states The given states
        """
        workspace.FeedBlob(self.input_blob, states)
        workspace.RunNetOnce(self._predict_model.net)
        return workspace.FetchBlob(self.output_blob)
