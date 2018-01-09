#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta
from caffe2.python import model_helper
from caffe2.python import workspace

from ml.rl.training.rl_predictor import RLPredictor
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet, MISSING_VALUE

import logging
logger = logging.getLogger(__name__)


class ContinuousActionDQNPredictor(RLPredictor):
    def predict(self, states, actions):
        """ Returns values for each state/action pair
        :param states states as list of feature -> value dict
        :param actions actions as list of feature -> value dict
        """

        examples = []
        for i in range(len(states)):
            examples.append({**states[i], **actions[i]})
        return RLPredictor.predict(self, examples)

    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def from_trainers(
        cls, trainer, state_features, action_features,
        state_normalization_parameters, action_normalization_parameters
    ):
        """ Creates DiscreteActionPredictor from a list of action trainers

        :param trainer DiscreteActionTrainer
        :param state_features list of state feature names
        :param action_features list of action feature names
        """
        # ensure state and action IDs have no intersection
        assert (len(set(state_features) & set(action_features)) == 0)
        normalization_parameters = dict(
            list(state_normalization_parameters.items()) +
            list(action_normalization_parameters.items())
        )
        features = state_features + action_features

        int_features = [int(feature) for feature in features]
        inputs = [
            'input/float_features.lengths', 'input/float_features.keys',
            'input/float_features.values'
        ]
        workspace.FeedBlob(
            'input/float_features.lengths', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys', np.zeros(1, dtype=np.int64)
        )
        workspace.FeedBlob(
            'input/float_features.values', np.zeros(1, dtype=np.float32)
        )
        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        dense_input = net.NextBlob('dense_input')
        default_input_value = net.NextBlob('default_input_value')
        net.GivenTensorFill(
            [], [default_input_value], shape=[], values=[MISSING_VALUE]
        )
        net.SparseToDenseMask(
            [
                'input/float_features.keys',
                'input/float_features.values',
                default_input_value,
                'input/float_features.lengths',
            ], [dense_input],
            mask=int_features
        )
        for i, feature in enumerate(features):
            net.Slice(
                [dense_input],
                [feature],
                starts=[0, i],
                ends=[-1, (i + 1)],
            )
        normalizer = PreprocessorNet(net, True)
        parameters = list(normalizer.parameters[:])
        normalized_input_blobs = []
        zero = "ZERO_from_trainers"
        workspace.FeedBlob(zero, np.array(0))
        parameters.append(zero)
        for feature in features:
            normalized_input_blob, blob_parameters = normalizer.preprocess_blob(
                feature,
                normalization_parameters[feature],
            )
            parameters.extend(blob_parameters)
            normalized_input_blobs.append(normalized_input_blob)

        concatenated_input_blob = "PredictorInput"
        output_dim = "PredictorOutputDim"
        for i, inp in enumerate(normalized_input_blobs):
            logger.info("input# {}: {}".format(i, inp))
        net.Concat(
            normalized_input_blobs, [concatenated_input_blob, output_dim],
            axis=1
        )
        net.NanCheck(concatenated_input_blob, concatenated_input_blob)

        q_lengths = "output/string_weighted_multi_categorical_features.values.lengths"
        workspace.FeedBlob(q_lengths, np.array([1], dtype=np.int32))
        q_keys = "output/string_weighted_multi_categorical_features.values.keys"
        workspace.FeedBlob(q_keys, np.array(['a']))
        q_values_matrix = net.NextBlob('q_values_matrix')
        trainer.build_predictor(model, concatenated_input_blob, q_values_matrix)
        parameters.extend(model.GetAllParams())

        q_values = 'output/string_weighted_multi_categorical_features.values.values'
        workspace.FeedBlob(q_values, np.array([1.0]))
        net.FlattenToVec([q_values_matrix], [q_values])
        net.ConstantFill(
            [q_values], [q_keys],
            value="Q",
            dtype=caffe2_pb2.TensorProto.STRING
        )
        net.ConstantFill(
            [q_values], [q_lengths],
            value=1,
            dtype=caffe2_pb2.TensorProto.INT32
        )

        q_feature_lengths = "output/string_weighted_multi_categorical_features.lengths"
        workspace.FeedBlob(q_feature_lengths, np.array([1], dtype=np.int32))
        net.ConstantFill(
            [q_values], [q_feature_lengths],
            value=1,
            dtype=caffe2_pb2.TensorProto.INT32
        )
        q_feature_keys = "output/string_weighted_multi_categorical_features.keys"
        workspace.FeedBlob(q_feature_keys, np.array([0], dtype=np.int64))
        net.ConstantFill(
            [q_values], [q_feature_keys],
            value=0,
            dtype=caffe2_pb2.TensorProto.INT64
        )

        output_blobs = [
            q_feature_lengths,
            q_feature_keys,
            q_lengths,
            q_keys,
            q_values,
        ]

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        predictor = cls(
            net, inputs, output_blobs, parameters, workspace.CurrentWorkspace()
        )
        return predictor
