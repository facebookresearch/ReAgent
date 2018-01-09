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


class DiscreteActionPredictor(RLPredictor):
    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def from_trainers(
        cls, trainer, features, actions, normalization_parameters
    ):
        """ Creates DiscreteActionPredictor from a list of action trainers

        :param trainer DiscreteActionTrainer
        :param features list of state feature names
        :param actions list of action names
        """
        int_features = [int(feature) for feature in features]
        inputs = [
            'input/float_features.lengths', 'input/float_features.keys',
            'input/float_features.values'
        ]
        workspace.FeedBlob(
            'input/float_features.lengths', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.values', np.zeros(1, dtype=np.float32)
        )
        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        dense_input = net.NextBlob('dense_input')
        workspace.FeedBlob(dense_input, np.zeros(1, dtype=np.float32))
        default_input_value = net.NextBlob('default_input_value')
        workspace.FeedBlob(
            default_input_value, np.array([MISSING_VALUE], dtype=np.float32)
        )
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
        parameters.append(default_input_value)
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

        q_values = "q_values"
        workspace.FeedBlob(q_values, np.zeros(1, dtype=np.float32))
        trainer.build_predictor(model, concatenated_input_blob, q_values)
        parameters.extend(model.GetAllParams())

        action_names = net.NextBlob("action_names")
        parameters.append(action_names)
        workspace.FeedBlob(action_names, np.array(actions))
        action_range = net.NextBlob("action_range")
        parameters.append(action_range)
        workspace.FeedBlob(action_range, np.array(list(range(len(actions)))))

        output_shape = net.NextBlob("output_shape")
        workspace.FeedBlob(output_shape, np.zeros(1, dtype=np.int64))
        net.Shape([q_values], [output_shape])
        output_shape_row_count = net.NextBlob("output_shape_row_count")
        net.Slice(
            [output_shape], [output_shape_row_count], starts=[0], ends=[1]
        )

        output_row_shape = net.NextBlob("output_row_shape")
        workspace.FeedBlob(output_row_shape, np.zeros(1, dtype=np.int64))
        net.Slice([q_values], [output_row_shape], starts=[0, 0], ends=[-1, 1])

        output_feature_keys = 'output/string_weighted_multi_categorical_features.keys'
        workspace.FeedBlob(output_feature_keys, np.zeros(1, dtype=np.int64))
        output_feature_keys_matrix = net.NextBlob('output_feature_keys_matrix')
        net.ConstantFill(
            [output_row_shape], [output_feature_keys_matrix],
            value=0,
            dtype=caffe2_pb2.TensorProto.INT64
        )
        net.FlattenToVec(
            [output_feature_keys_matrix],
            [output_feature_keys],
        )

        output_feature_lengths = \
            'output/string_weighted_multi_categorical_features.lengths'
        workspace.FeedBlob(output_feature_lengths, np.zeros(1, dtype=np.int32))
        output_feature_lengths_matrix = net.NextBlob(
            'output_feature_lengths_matrix'
        )
        net.ConstantFill(
            [output_row_shape], [output_feature_lengths_matrix],
            value=1,
            dtype=caffe2_pb2.TensorProto.INT32
        )
        net.FlattenToVec(
            [output_feature_lengths_matrix],
            [output_feature_lengths],
        )

        output_keys = 'output/string_weighted_multi_categorical_features.values.keys'
        workspace.FeedBlob(output_keys, np.array(['a']))
        net.Tile([action_names, output_shape_row_count], [output_keys], axis=1)

        output_lengths_matrix = net.NextBlob('output_lengths_matrix')
        net.ConstantFill(
            [output_row_shape], [output_lengths_matrix],
            value=len(actions),
            dtype=caffe2_pb2.TensorProto.INT32
        )
        output_lengths = \
            'output/string_weighted_multi_categorical_features.values.lengths'
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        net.FlattenToVec(
            [output_lengths_matrix],
            [output_lengths],
        )

        output_values = \
            'output/string_weighted_multi_categorical_features.values.values'
        workspace.FeedBlob(output_values, np.array([1.0]))
        net.FlattenToVec([q_values], [output_values])

        output_blobs = [
            output_feature_keys,
            output_feature_lengths,
            output_keys,
            output_lengths,
            output_values,
        ]
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        predictor = cls(
            net, inputs, output_blobs, parameters, workspace.CurrentWorkspace()
        )
        return predictor
