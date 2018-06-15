#!/usr/bin/env python3

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor.predictor_exporter import (
    load_from_db,
    PredictorExportMeta,
    prepare_prediction_net,
    save_to_db,
)
from caffe2.python import core, model_helper, workspace
from caffe2.python.predictor_constants import predictor_constants
from caffe2.python.predictor.predictor_py_utils import GetBlobs

from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet

import logging

logger = logging.getLogger(__name__)


class DDPGPredictor(object):
    def __init__(self, net, parameters, int_features) -> None:
        self._net = net
        self._input_blobs = [
            "input/float_features.lengths",
            "input/float_features.keys",
            "input/float_features.values",
        ]
        if int_features:
            self._input_blobs.extend(
                [
                    "input/int_features.lengths",
                    "input/int_features.keys",
                    "input/int_features.values",
                ]
            )
        self._output_blobs = [
            "output/float_features.lengths",
            "output/float_features.keys",
            "output/float_features.values",
        ]
        self._parameters = parameters

    def policy(self):
        """TODO: Return actions when exporting final net and fill in this
        function."""
        pass

    def actor_prediction(self, float_state_features, int_state_features=None):
        """ Actor Prediction - Returns action for each float_feature state. Also,
        accepts int_feature states.

        :param float_state_features A list of feature -> float value dict examples
        :param int_features A list of feature -> int value dict examples
        """
        workspace.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_state_features], dtype=np.int32),
        )
        workspace.FeedBlob(
            "input/float_features.keys",
            np.array(
                [list(e.keys()) for e in float_state_features], dtype=np.int64
            ).flatten(),
        )
        workspace.FeedBlob(
            "input/float_features.values",
            np.array(
                [list(e.values()) for e in float_state_features], dtype=np.float32
            ).flatten(),
        )

        if int_state_features:
            workspace.FeedBlob(
                "input/int_features.lengths",
                np.array([len(e) for e in int_state_features], dtype=np.int32),
            )
            workspace.FeedBlob(
                "input/int_features.keys",
                np.array(
                    [list(e.keys()) for e in int_state_features], dtype=np.int64
                ).flatten(),
            )
            workspace.FeedBlob(
                "input/int_features.values",
                np.array(
                    [list(e.values()) for e in int_state_features], dtype=np.int32
                ).flatten(),
            )

        workspace.RunNet(self._net)

        results = workspace.FetchBlob("output/float_features.values")
        return results

    def critic_prediction(self, float_state_features, int_state_features, actions):
        """ Critic Prediction - Returns values for each state/action pair. Accepts
        int_features as 3rd optional parameter.

        :param float_state_features states as list of feature -> float value dict
        :param int_state_features states as list of feature -> int value dict
        :param actions actions as list of feature -> value dict
        """
        float_examples = []
        for i in range(len(float_state_features)):
            float_examples.append({**float_state_features[i], **actions[i]})

        workspace.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_examples], dtype=np.int32),
        )
        workspace.FeedBlob(
            "input/float_features.keys",
            np.array(
                [list(e.keys()) for e in float_examples], dtype=np.int64
            ).flatten(),
        )
        workspace.FeedBlob(
            "input/float_features.values",
            np.array(
                [list(e.values()) for e in float_examples], dtype=np.float32
            ).flatten(),
        )

        if int_state_features is not None:
            workspace.FeedBlob(
                "input/int_features.lengths",
                np.array([len(e) for e in int_state_features], dtype=np.int32),
            )
            workspace.FeedBlob(
                "input/int_features.keys",
                np.array(
                    [list(e.keys()) for e in int_state_features], dtype=np.int64
                ).flatten(),
            )
            workspace.FeedBlob(
                "input/int_features.values",
                np.array(
                    [list(e.values()) for e in int_state_features], dtype=np.int32
                ).flatten(),
            )

        workspace.RunNet(self._net)

        results = workspace.FetchBlob("output/float_features.values")
        return results

    def get_predictor_export_meta(self):
        """
        Returns a PredictorExportMeta object
        """
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    def save(self, db_path, db_type):
        """ Saves network to db

        :param db_path see save_to_db
        :param db_type see save_to_db
        """
        meta = self.get_predictor_export_meta()
        for parameter in self._parameters:
            parameter_data = workspace.FetchBlob(parameter)
            logger.info("DATA TYPE " + parameter_data.dtype.kind)
            if parameter_data.dtype.kind in {"U", "S", "O"}:
                continue  # Don't bother checking string blobs for nan
            logger.info("Checking parameter {} for nan".format(parameter))
            if np.any(np.isnan(parameter_data)):
                logger.info("WARNING: parameter {} is nan".format(parameter))
        save_to_db(db_type, db_path, meta)

    @classmethod
    def load(cls, db_path, db_type, int_features=False):
        """ Creates Predictor by loading from a database

        :param db_path see load_from_db
        :param db_type see load_from_db
        :param int_features bool indicating if int_features are present
        """
        net = prepare_prediction_net(db_path, db_type)
        meta = load_from_db(db_path, db_type)
        parameters = GetBlobs(meta, predictor_constants.PARAMETERS_BLOB_TYPE)
        return cls(net, parameters, int_features)

    @classmethod
    def export_actor(
        cls,
        trainer,
        state_normalization_parameters,
        int_features=False,
        model_on_gpu=False,
    ):
        """Export caffe2 preprocessor net and pytorch actor forward pass as one
        caffe2 net.

        :param trainer DDPGTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param int_features boolean indicating if int features blob will be present
        """
        input_dim = trainer.state_dim
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.actor, input_dim, model_on_gpu
        )
        actor_input_blob, actor_output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
            buffer
        )
        torch_workspace = caffe2_netdef.workspace

        parameters = []
        for blob_str in torch_workspace.Blobs():
            workspace.FeedBlob(blob_str, torch_workspace.FetchBlob(blob_str))
            parameters.append(blob_str)

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)

        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        C2.set_model(model)

        workspace.FeedBlob("input/float_features.lengths", np.zeros(1, dtype=np.int32))
        workspace.FeedBlob("input/float_features.keys", np.zeros(1, dtype=np.int64))
        workspace.FeedBlob("input/float_features.values", np.zeros(1, dtype=np.float32))

        input_feature_lengths = "input_feature_lengths"
        input_feature_keys = "input_feature_keys"
        input_feature_values = "input_feature_values"

        if int_features:
            workspace.FeedBlob(
                "input/int_features.lengths", np.zeros(1, dtype=np.int32)
            )
            workspace.FeedBlob("input/int_features.keys", np.zeros(1, dtype=np.int64))
            workspace.FeedBlob("input/int_features.values", np.zeros(1, dtype=np.int32))
            C2.net().Cast(
                ["input/int_features.values"],
                ["input/int_features.values_float"],
                dtype=caffe2_pb2.TensorProto.FLOAT,
            )
            C2.net().MergeMultiScalarFeatureTensors(
                [
                    "input/float_features.lengths",
                    "input/float_features.keys",
                    "input/float_features.values",
                    "input/int_features.lengths",
                    "input/int_features.keys",
                    "input/int_features.values_float",
                ],
                [input_feature_lengths, input_feature_keys, input_feature_values],
            )
        else:
            C2.net().Copy(["input/float_features.lengths"], [input_feature_lengths])
            C2.net().Copy(["input/float_features.keys"], [input_feature_keys])
            C2.net().Copy(["input/float_features.values"], [input_feature_values])

        preprocessor = PreprocessorNet(net, True)
        parameters.extend(preprocessor.parameters)
        state_normalized_dense_matrix, new_parameters = preprocessor.normalize_sparse_matrix(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            state_normalization_parameters,
            "state_norm",
            False,
            False,
        )
        parameters.extend(new_parameters)
        net.Copy([state_normalized_dense_matrix], [actor_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_init_net)
        net.AppendNet(torch_predict_net)

        C2.FlattenToVec(C2.ArgMax(actor_output_blob))
        output_lengths = "output/float_features.lengths"
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [C2.FlattenToVec(C2.ArgMax(actor_output_blob))],
            [output_lengths],
            value=trainer.actor.layers[-1].out_features,
            dtype=caffe2_pb2.TensorProto.INT32,
        )

        output_keys = "output/float_features.keys"
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int32))
        C2.net().LengthsRangeFill([output_lengths], [output_keys])

        output_values = "output/float_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([actor_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, parameters, int_features)

    @classmethod
    def export_critic(
        cls,
        trainer,
        state_normalization_parameters,
        action_normalization_parameters,
        int_features=False,
        model_on_gpu=False,
    ):
        """Export caffe2 preprocessor net and pytorch critic forward pass as one
        caffe2 net.

        :param trainer DDPGTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param action_normalization_parameters action NormalizationParameters
        :param int_features boolean indicating if int features blob will be present
        """
        input_dim = trainer.state_dim + trainer.action_dim
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.critic, input_dim, model_on_gpu
        )
        critic_input_blob, critic_output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
            buffer
        )
        torch_workspace = caffe2_netdef.workspace

        parameters = []
        for blob_str in torch_workspace.Blobs():
            workspace.FeedBlob(blob_str, torch_workspace.FetchBlob(blob_str))
            parameters.append(blob_str)

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)

        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        C2.set_model(model)

        workspace.FeedBlob("input/float_features.lengths", np.zeros(1, dtype=np.int32))
        workspace.FeedBlob("input/float_features.keys", np.zeros(1, dtype=np.int64))
        workspace.FeedBlob("input/float_features.values", np.zeros(1, dtype=np.float32))

        input_feature_lengths = "input_feature_lengths"
        input_feature_keys = "input_feature_keys"
        input_feature_values = "input_feature_values"

        if int_features:
            workspace.FeedBlob(
                "input/int_features.lengths", np.zeros(1, dtype=np.int32)
            )
            workspace.FeedBlob("input/int_features.keys", np.zeros(1, dtype=np.int64))
            workspace.FeedBlob("input/int_features.values", np.zeros(1, dtype=np.int32))
            C2.net().Cast(
                ["input/int_features.values"],
                ["input/int_features.values_float"],
                dtype=caffe2_pb2.TensorProto.FLOAT,
            )
            C2.net().MergeMultiScalarFeatureTensors(
                [
                    "input/float_features.lengths",
                    "input/float_features.keys",
                    "input/float_features.values",
                    "input/int_features.lengths",
                    "input/int_features.keys",
                    "input/int_features.values_float",
                ],
                [input_feature_lengths, input_feature_keys, input_feature_values],
            )
        else:
            C2.net().Copy(["input/float_features.lengths"], [input_feature_lengths])
            C2.net().Copy(["input/float_features.keys"], [input_feature_keys])
            C2.net().Copy(["input/float_features.values"], [input_feature_values])

        preprocessor = PreprocessorNet(net, True)
        parameters.extend(preprocessor.parameters)
        state_normalized_dense_matrix, new_parameters = preprocessor.normalize_sparse_matrix(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            state_normalization_parameters,
            "state_norm",
            False,
            False,
        )
        parameters.extend(new_parameters)
        action_normalized_dense_matrix, new_parameters = preprocessor.normalize_sparse_matrix(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            action_normalization_parameters,
            "action_norm",
            False,
            False,
        )
        parameters.extend(new_parameters)
        state_action_normalized = "state_action_normalized"
        state_action_normalized_dim = "state_action_normalized_dim"
        net.Concat(
            [state_normalized_dense_matrix, action_normalized_dense_matrix],
            [state_action_normalized, state_action_normalized_dim],
            axis=1,
        )
        net.Copy([state_action_normalized], [critic_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_init_net)
        net.AppendNet(torch_predict_net)

        C2.FlattenToVec(C2.ArgMax(critic_output_blob))
        output_lengths = "output/float_features.lengths"
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [C2.FlattenToVec(C2.ArgMax(critic_output_blob))],
            [output_lengths],
            value=trainer.critic.layers[-1].out_features,
            dtype=caffe2_pb2.TensorProto.INT32,
        )

        output_keys = "output/float_features.keys"
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int32))
        C2.net().LengthsRangeFill([output_lengths], [output_keys])

        output_values = "output/float_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([critic_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, parameters, int_features)
