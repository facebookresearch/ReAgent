from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np

from ml.rl.training.ml_trainer import MLTrainer, MakeForwardPassOps, GenerateLossOps

# @build:deps [
# @/caffe2/caffe2/python:caffe2_py
# ]

from caffe2.python import workspace


def MakeFowardPassScaledOps(
    model, model_id, output_blob, external_blob, external_output_blob
):
    scale_output_beforeagg = model_id + "_tmp_output"
    model.net.Mul([output_blob, external_blob], scale_output_beforeagg)
    model.net.ReduceBackSum(scale_output_beforeagg, external_output_blob)


def GenerateLossScaledOps(
    model, model_id, external_output_blob, label_blob, loss_blob
):
    dimextend_extoutput_blob = model.net.ExpandDims(
        external_output_blob, dims=[1]
    )
    GenerateLossOps(
        model, model_id, dimextend_extoutput_blob, label_blob, loss_blob
    )


def MakeFowardPassExtensionOps(
    model, model_id, output_blob, external_blob, external_output_blob,
    extension_mltrainer
):
    concat_input = model_id + "_concatenated_input"
    model.net.Concat(
        [external_blob, output_blob],
        [concat_input, concat_input + "_concat_dims"],
        axis=1
    )
    MakeForwardPassOps(
        model, model_id + "_ext", concat_input, external_output_blob,
        extension_mltrainer.weights, extension_mltrainer.biases,
        extension_mltrainer.activations, extension_mltrainer.layers,
        extension_mltrainer.dropout_ratio, False
    )


def GenerateLossExtensionOps(
    model, model_id, external_output_blob, label_blob, loss_blob
):
    # This loss is computed on external_output after output (not single dim)
    # is aggregated with another vector that might be of different length,
    # then use another network's forward to generate value
    # This is to support for arctor-critic

    # (1) here the loss is not MSE, but the value itself? -q_pred to maximize q
    inv_output = model_id + "_ext" + "+_output_inverse"
    model.net.Scale(external_output_blob, inv_output, scale=-1.0)
    model.net.AveragedLoss([inv_output], [loss_blob])
    # # (2) positive loss when label > q_pred, trying to maximize q
    # dist = net.Sub([label_blob, external_output_blob], model_id + "dist")
    # dist = net.Relu(dist, dist)
    # loss_blob = dist.AveragedLoss([], [loss_blob])
    ## (3) L1 Distance? to increase Q value?:
    # dist = net.L1Distance([label_blob, external_output_blob], model_id + "dist")
    # loss_blob = dist.AveragedLoss([], [loss_blob])


class MLTrainerIP(MLTrainer):
    """ This is meant to be a inner-product topped extension for current ml
    """

    def __init__(self, name, parameters, scaled_output=True):
        """

        :param name: A unique name for this trainer used to create the data on the
            caffe2 workspace
        :param layers: A list of integers describing the layer sizes
        :param activations: A list of strings describing the activation functions
        """
        self.scaled_output = scaled_output
        MLTrainer.__init__(
            self,
            name,
            parameters
        )

    def _setup_initial_blobs(self):
        MLTrainer._setup_initial_blobs(self)
        if self.scaled_output:
            self._setup_initial_extension_blobs()

    def _build_fwd_pass_train_model(self):
        self.train_model.StopGradient(self.labels_blob, self.labels_blob)
        MakeForwardPassOps(
            self.train_model, self.model_id + "_train", self.input_blob,
            self.output_blob, self.weights, self.biases, self.activations,
            self.layers, self.dropout_ratio
        )
        if self.scaled_output:
            MakeFowardPassScaledOps(
                self.train_model, self.model_id + "_train", self.output_blob,
                self.external_blob, self.external_output_blob
            )

    def _generate_train_model_loss(self):
        if self.scaled_output:
            GenerateLossScaledOps(
                self.train_model, self.model_id + "_train",
                self.external_output_blob, self.labels_blob, self.loss_blob
            )
        else:
            GenerateLossOps(
                self.train_model, self.model_id + "_train", self.output_blob,
                self.labels_blob, self.loss_blob
            )

    # also will be used by extension
    def _setup_initial_extension_blobs(self):
        self.external_blob = "ModeExt_" + self.model_id
        workspace.FeedBlob(self.external_blob, np.zeros(1, dtype=np.float32))

        self.external_output_blob = "ModeOutputExt_" + self.model_id
        workspace.FeedBlob(
            self.external_output_blob, np.zeros(1, dtype=np.float32)
        )

    # bundle for train with scaled vector and inner product output
    def build_predictor_ext(
        self, model, input_blob, output_blob, external_blob,
        external_output_blob
    ):
        MakeForwardPassOps(
            model, self.model_id + "_score", input_blob, output_blob,
            self.weights, self.biases, self.activations, self.layers,
            self.dropout_ratio, True
        )

        MakeFowardPassScaledOps(
            model, self.model_id + "_score", output_blob, external_blob,
            external_output_blob
        )
        return self.weights + self.biases

    def extend_predictor_only(
        self, model, output_blob, external_blob, external_output_blob
    ):
        # no parameter involved so no return
        MakeFowardPassScaledOps(
            model, self.model_id + "_score", output_blob, external_blob,
            external_output_blob
        )

    def score_wexternal(self, outputs, scale):
        assert (outputs.shape == scale.shape)
        return np.sum(outputs * scale, axis=1)

    def train_wexternal(self, inputs, labels, external, evaluate=False):
        workspace.FeedBlob(self.input_blob, inputs)
        workspace.FeedBlob(self.labels_blob, labels)
        workspace.FeedBlob(self.external_blob, external)
        workspace.RunNetOnce(self.train_model.net)
        if evaluate:
            return (
                np.squeeze(workspace.FetchBlob(self.external_output_blob)),
                self.loss
            )

    @property
    def external_output(self):
        return workspace.FetchBlob(self.external_output_blob)


class MLTrainerExt(MLTrainerIP):
    def __init__(
        self, name, parameters, scaled_output=False, extension_mltrainer=None
    ):
        """

        :param name: A unique name for this trainer used to create the data on the
            caffe2 workspace
        :param layers: A list of integers describing the layer sizes.
        :param activations: A list of strings describing the activation functions.
        """
        self.extension_mltrainer = extension_mltrainer
        MLTrainerIP.__init__(self, name, parameters, scaled_output)

    def _setup_initial_blobs(self):
        MLTrainer._setup_initial_blobs(self)
        if self.extension_mltrainer is not None or self.scaled_output:
            self._setup_initial_extension_blobs()

    def _build_fwd_pass_train_model(self):
        self.train_model.StopGradient(self.labels_blob, self.labels_blob)
        MakeForwardPassOps(
            self.train_model, self.model_id + "_train", self.input_blob,
            self.output_blob, self.weights, self.biases, self.activations,
            self.layers, self.dropout_ratio
        )
        if self.extension_mltrainer is not None:
            MakeFowardPassExtensionOps(
                self.train_model, self.model_id + "_train", self.output_blob,
                self.external_blob, self.external_output_blob,
                self.extension_mltrainer
            )
        elif self.scaled_output:
            MakeFowardPassScaledOps(
                self.train_model, self.model_id + "_train", self.output_blob,
                self.external_blob, self.external_output_blob
            )

    def _generate_train_model_loss(self):
        if self.extension_mltrainer is not None:
            GenerateLossExtensionOps(
                self.train_model, self.model_id + "_train",
                self.external_output_blob, self.labels_blob, self.loss_blob
            )
        elif self.scaled_output:
            GenerateLossScaledOps(
                self.train_model, self.model_id + "_train",
                self.external_output_blob, self.labels_blob, self.loss_blob
            )
        else:
            GenerateLossOps(
                self.train_model, self.model_id + "_train", self.output_blob,
                self.labels_blob, self.loss_blob
            )

    # bundle for extension_net
    def build_predictor_ext(
        self, model, input_blob, output_blob, external_blob,
        external_output_blob, extension_mltrainer
    ):
        MakeForwardPassOps(
            model, self.model_id + "_score", input_blob, output_blob,
            self.weights, self.biases, self.activations, self.layers,
            self.dropout_ratio, True
        )

        MakeFowardPassExtensionOps(
            model, self.model_id + "_score", output_blob, external_blob,
            external_output_blob, extension_mltrainer
        )
        return self.weights + self.biases

    def extend_predictor_only(
        self, model, output_blob, external_blob, external_output_blob,
        extension_mltrainer
    ):
        # no parameter involved so no return
        MakeFowardPassExtensionOps(
            model, self.model_id + "_score", output_blob, external_blob,
            external_output_blob, extension_mltrainer
        )

    def score_wexternal(self, outputs, external_input):
        concat_input = np.concatenate([external_input, outputs], axis=1)
        return self.extension_mltrainer.score(concat_input).flatten()
