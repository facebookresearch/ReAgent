#!/usr/bin/env python3

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor.predictor_exporter import \
    load_from_db, PredictorExportMeta, prepare_prediction_net, save_to_db
from caffe2.python import core, model_helper, workspace
from caffe2.python.predictor_constants import predictor_constants
from caffe2.python.predictor.predictor_py_utils import GetBlobs

from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet

import logging
logger = logging.getLogger(__name__)


class DDPGPredictor(object):
    def __init__(self, net, parameters) -> None:
        self._net = net
        self._input_blobs = [
            'input/float_features.lengths',
            'input/float_features.keys',
            'input/float_features.values',
        ]
        self._output_blobs = [
            'output/float_features.lengths',
            'output/float_features.keys',
            'output/float_features.values',
        ]
        self._parameters = parameters

    def actor_prediction(self, states):
        """ Actor Prediction - Returns action for each state
        :param states states as list of feature -> value dict
        """
        examples = []
        for i in range(len(states)):
            examples.append({**states[i]})

        workspace.FeedBlob(
            'input/float_features.lengths',
            np.array([len(e) for e in examples], dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys',
            np.array([list(e.keys()) for e in examples],
                     dtype=np.int32).flatten()
        )
        workspace.FeedBlob(
            'input/float_features.values',
            np.array([list(e.values()) for e in examples],
                     dtype=np.float32).flatten()
        )
        workspace.RunNet(self._net)

        results = workspace.FetchBlob('output/float_features.values')
        return results

    def critic_prediction(self, states, actions):
        """ Critic Prediction - Returns values for each state/action pair
        :param states states as list of feature -> value dict
        :param actions actions as list of feature -> value dict
        """
        examples = []
        for i in range(len(states)):
            examples.append({**states[i], **actions[i]})

        workspace.FeedBlob(
            'input/float_features.lengths',
            np.array([len(e) for e in examples], dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys',
            np.array([list(e.keys()) for e in examples],
                     dtype=np.int32).flatten()
        )
        workspace.FeedBlob(
            'input/float_features.values',
            np.array([list(e.values()) for e in examples],
                     dtype=np.float32).flatten()
        )
        workspace.RunNet(self._net)

        results = workspace.FetchBlob('output/float_features.values')
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
            if parameter_data.dtype.kind in {'U', 'S', 'O'}:
                continue  # Don't bother checking string blobs for nan
            logger.info("Checking parameter {} for nan".format(parameter))
            if np.any(np.isnan(parameter_data)):
                logger.info("WARNING: parameter {} is nan".format(parameter))
        save_to_db(db_type, db_path, meta)

    @classmethod
    def load(cls, db_path, db_type):
        """ Creates Predictor by loading from a database

        :param db_path see load_from_db
        :param db_type see load_from_db
        """
        net = prepare_prediction_net(db_path, db_type)
        meta = load_from_db(db_path, db_type)
        parameters = GetBlobs(meta, predictor_constants.PARAMETERS_BLOB_TYPE)
        return cls(net, parameters)

    @classmethod
    def export_actor(cls, trainer, state_normalization_parameters):
        """Export caffe2 preprocessor net and pytorch actor forward pass as one
        caffe2 net.
        """
        input_dim = len(state_normalization_parameters)
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.actor, input_dim
        )
        actor_input_blob, actor_output_blob, caffe2_netdef =\
            PytorchCaffe2Converter.buffer_to_caffe2_netdef(buffer)
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

        workspace.FeedBlob(
            'input/float_features.lengths', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.values', np.zeros(1, dtype=np.float32)
        )

        preprocessor = PreprocessorNet(net, True)
        parameters.extend(preprocessor.parameters)
        state_normalized_dense_matrix, new_parameters = \
            preprocessor.normalize_sparse_matrix(
                'input/float_features.lengths',
                'input/float_features.keys',
                'input/float_features.values',
                state_normalization_parameters,
                'state_norm',
            )
        parameters.extend(new_parameters)
        net.Copy([state_normalized_dense_matrix], [actor_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_init_net)
        net.AppendNet(torch_predict_net)

        C2.FlattenToVec(C2.ArgMax(actor_output_blob))
        output_lengths = 'output/float_features.lengths'
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [C2.FlattenToVec(C2.ArgMax(actor_output_blob))], [output_lengths],
            value=trainer.actor.layers[-1].out_features,
            dtype=caffe2_pb2.TensorProto.INT32
        )

        output_keys = 'output/float_features.keys'
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int32))
        C2.net().LengthsRangeFill([output_lengths], [output_keys])

        output_values = 'output/float_features.values'
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([actor_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, parameters)

    @classmethod
    def export_critic(
        cls, trainer, state_normalization_parameters,
        action_normalization_parameters
    ):
        """Export caffe2 preprocessor net and pytorch critic forward pass as one
        caffe2 net.
        """
        input_dim =\
            len(state_normalization_parameters) + len(action_normalization_parameters)
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.critic, input_dim
        )
        critic_input_blob, critic_output_blob, caffe2_netdef =\
            PytorchCaffe2Converter.buffer_to_caffe2_netdef(buffer)
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

        workspace.FeedBlob(
            'input/float_features.lengths', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.values', np.zeros(1, dtype=np.float32)
        )

        preprocessor = PreprocessorNet(net, True)
        parameters.extend(preprocessor.parameters)
        state_normalized_dense_matrix, new_parameters = \
            preprocessor.normalize_sparse_matrix(
                'input/float_features.lengths',
                'input/float_features.keys',
                'input/float_features.values',
                state_normalization_parameters,
                'state_norm',
            )
        parameters.extend(new_parameters)
        action_normalized_dense_matrix, new_parameters = \
            preprocessor.normalize_sparse_matrix(
                'input/float_features.lengths',
                'input/float_features.keys',
                'input/float_features.values',
                action_normalization_parameters,
                'action_norm',
            )
        parameters.extend(new_parameters)
        state_action_normalized = 'state_action_normalized'
        state_action_normalized_dim = 'state_action_normalized_dim'
        net.Concat(
            [state_normalized_dense_matrix, action_normalized_dense_matrix],
            [state_action_normalized, state_action_normalized_dim],
            axis=1
        )
        net.Copy([state_action_normalized], [critic_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_init_net)
        net.AppendNet(torch_predict_net)

        C2.FlattenToVec(C2.ArgMax(critic_output_blob))
        output_lengths = 'output/float_features.lengths'
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [C2.FlattenToVec(C2.ArgMax(critic_output_blob))], [output_lengths],
            value=trainer.critic.layers[-1].out_features,
            dtype=caffe2_pb2.TensorProto.INT32
        )

        output_keys = 'output/float_features.keys'
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int32))
        C2.net().LengthsRangeFill([output_lengths], [output_keys])

        output_values = 'output/float_features.values'
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([critic_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, parameters)
