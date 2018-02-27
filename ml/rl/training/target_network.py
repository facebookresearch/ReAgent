#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

import numpy as np

# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# ]

from caffe2.python import model_helper, workspace, core

from ml.rl.caffe_utils import C2
from ml.rl.training.ml_trainer import MakeForwardPassOps
from ml.rl.training.conv.ml_conv_trainer import MLConvTrainer, MakeConvPassOps


class TargetNetwork(object):
    """ The target network is used to compute the labels in deep TD learning.
        This class computes the labels and updates the weights of the target
        network
    """

    def __init__(self, trainer, target_update_rate):
        self._trainer = trainer
        self._target_update_rate = target_update_rate
        self.enabled_slow_updates = False
        self.tn_model_id = self._trainer.model_id + "_target"
        self._is_conv_tn = isinstance(self._trainer, MLConvTrainer)

        self._setup_initial_blobs()
        self._setup_predict_net()
        self._setup_update_net()

        workspace.RunNetOnce(self._predict_model.param_init_net)
        workspace.CreateNet(self._predict_model.net)
        workspace.RunNetOnce(self._update_model.param_init_net)
        workspace.CreateNet(self._update_model.net)

    def _setup_initial_blobs(self):
        self._update_rate_blob = self.tn_model_id + "_update_rate"
        workspace.FeedBlob(
            self._update_rate_blob, np.array([1], dtype=np.float32)
        )
        self._retain_rate_blob = self.tn_model_id + "_retain_rate"
        workspace.FeedBlob(
            self._retain_rate_blob, np.array([0], dtype=np.float32)
        )
        self.input_blob = "TargetInput_" + self.tn_model_id
        workspace.FeedBlob(self.input_blob, np.zeros(1, dtype=np.float32))
        self.output_blob = "TargetOutput_" + self.tn_model_id
        workspace.FeedBlob(self.output_blob, np.zeros(1, dtype=np.float32))

        if self._is_conv_tn:
            self.output_conv_blob = "TargetConvOutput_" + self.tn_model_id
            workspace.FeedBlob(
                self.output_conv_blob, np.zeros(1, dtype=np.float32)
            )

        self._weights = self._copy_params(self._trainer.weights)
        self._biases = self._copy_params(self._trainer.biases)

        if self._is_conv_tn:
            self._conv_weights = self._copy_params(self._trainer.conv_weights)
            self._conv_biases = self._copy_params(self._trainer.conv_biases)

        self._update_model = model_helper.ModelHelper(
            name="TargetUpdateModel_" + self.tn_model_id
        )
        self._predict_model = model_helper.ModelHelper(
            name="TargetPredictModel_" + self.tn_model_id
        )

    def _setup_predict_net(self):
        fc_input_blob = self.input_blob

        if self._is_conv_tn:
            MakeConvPassOps(
                self._predict_model, self.tn_model_id, self._trainer.dims,
                self._trainer.conv_height_kernels,
                self._trainer.conv_width_kernels,
                self._trainer.pool_kernels_strides, self._trainer.pool_types,
                self.input_blob, self.output_conv_blob, self._conv_weights,
                self._conv_biases
            )
            fc_input_blob = self.output_conv_blob

        MakeForwardPassOps(
            self._predict_model, self.tn_model_id, fc_input_blob,
            self.output_blob, self._weights, self._biases,
            self._trainer.activations, self._trainer.layers,
            self._trainer.dropout_ratio, True
        )

    def _setup_update_net(self):
        self._add_update_ops(self._trainer.weights, self._weights)
        self._add_update_ops(self._trainer.biases, self._biases)

        if self._is_conv_tn:
            self._add_update_ops(self._trainer.conv_weights, self._conv_weights)
            self._add_update_ops(self._trainer.conv_biases, self._conv_biases)

    def _copy_params(self, params):
        copied_params = [param + "_target" for param in params]
        for target_param, source_param in zip(copied_params, params):
            workspace.FeedBlob(target_param, workspace.FetchBlob(source_param))
        return copied_params

    def _add_update_ops(self, source_params, target_params):
        for source_param, target_param in zip(source_params, target_params):
            self._update_model.net.WeightedSum(
                [
                    target_param, self._retain_rate_blob, source_param,
                    self._update_rate_blob
                ], [target_param]
            )

    def enable_slow_updates(self):
        if not self.enabled_slow_updates:
            workspace.FeedBlob(
                self._update_rate_blob,
                np.array([self._target_update_rate], dtype=np.float32)
            )
            workspace.FeedBlob(
                self._retain_rate_blob,
                np.array([1 - self._target_update_rate], dtype=np.float32)
            )
            self.enabled_slow_updates = True

    def target_update(self):
        """ Updates the weights of the target network according to the
            target_update_rate
        """
        workspace.RunNet(self._update_model.net)

    def target_values(self, input_blob: str) -> str:
        """ Estimates the values for the given inputs using the target network

        :param inputs The given inputs
        """
        output_blob = C2.NextBlob("output_blob")
        MakeForwardPassOps(
            C2.model(),
            self.tn_model_id,
            input_blob,
            output_blob,
            self._weights,
            self._biases,
            self._trainer.activations,
            self._trainer.layers,
            self._trainer.dropout_ratio,
            True,
        )
        return output_blob

    def target_values_concat(self, inputs):
        """ Estimates the values for the given inputs using the target network

        :param inputs The given inputs
        """
        self._concat_inputs(inputs)
        workspace.RunNet(self._predict_model.net)
        return workspace.FetchBlob(self.output_blob)

    def _concat_inputs(self, inputs: List[np.ndarray]) -> None:
        blobs_to_concat = []
        for i, input in enumerate(inputs):
            blob_name = self.input_blob + "_part_" + str(i)
            workspace.FeedBlob(blob_name, input)
            blobs_to_concat.append(blob_name)
        split_info = 'dummy_split_info'
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'Concat',
                blobs_to_concat,
                [self.input_blob, split_info],
                axis=1,
            )
        )
