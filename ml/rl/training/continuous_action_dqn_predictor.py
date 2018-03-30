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

from ml.rl.caffe_utils import C2
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.training.rl_predictor import RLPredictor

import logging
logger = logging.getLogger(__name__)


class ContinuousActionDQNPredictor(RLPredictor):
    def __init__(self, net, parameters):
        RLPredictor.__init__(self, net, parameters)
        self.is_discrete = False
        self._output_blobs.extend([
            'output/int_single_categorical_features.keys',
            'output/int_single_categorical_features.lengths',
            'output/int_single_categorical_features.values',
        ])

    def predict(self, states, actions):
        """ Returns values for each state/action pair
        :param states states as list of feature -> value dict
        :param actions actions as list of feature -> value dict
        """

        examples = []
        for i in range(len(states)):
            examples.append({**states[i], **actions[i]})
        return RLPredictor.predict(self, examples)

    def policy(self, states, actions):
        examples = []
        for i in range(len(states)):
            examples.append({**states[i], **actions[i]})
        return RLPredictor.policy(self, examples)

    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def export(
        cls,
        trainer,
        state_normalization_parameters,
        action_normalization_parameters,
    ):
        """ Creates ContinuousActionDQNPredictor from a list of action trainers

        :param trainer ContinuousActionDQNPredictor
        :param state_features list of state feature names
        :param action_features list of action feature names
        """
        # ensure state and action IDs have no intersection
        assert (
            len(
                set(state_normalization_parameters.keys()) &
                set(action_normalization_parameters.keys())
            ) == 0
        )

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
        parameters = []
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
        new_parameters, q_values = RLPredictor._forward_pass(
            model,
            trainer,
            state_action_normalized,
            ['Q'],
        )
        parameters.extend(new_parameters)

        flat_q_values_key = \
            'output/string_weighted_multi_categorical_features.values.values'
        num_examples, _ = C2.Reshape(C2.Size(flat_q_values_key), shape=[1])
        q_value_blob, _ = C2.Reshape(flat_q_values_key, shape=[1, -1])

        # Get 1 x n (number of examples) action index tensor under the max_q policy
        max_q_act_idxs = 'max_q_policy_actions'
        C2.net().FlattenToVec([C2.ArgMax(q_value_blob)], [max_q_act_idxs])
        max_q_act_blob = C2.Tile(max_q_act_idxs, num_examples, axis=0)

        # Get 1 x n (number of examples) action index tensor under the softmax policy
        temperature = C2.NextBlob("temperature")
        parameters.append(temperature)
        workspace.FeedBlob(
            temperature, np.array([trainer.rl_temperature], dtype=np.float32))
        tempered_q_values = C2.Div(q_value_blob, "temperature", broadcast=1)
        softmax_values = C2.Softmax(tempered_q_values)
        softmax_act_idxs_nested = 'softmax_act_idxs_nested'
        C2.net().WeightedSample([softmax_values], [softmax_act_idxs_nested])
        softmax_act_blob = C2.Tile(
            C2.FlattenToVec(softmax_act_idxs_nested),
            num_examples, axis=0)

        # Concat action idx vecs to get 2 x n tensor [[a_maxq, ..], [a_softmax, ..]]
        # transpose & flatten to get [a_maxq, a_softmax, a_maxq, a_softmax, ...]
        max_q_act_blob = C2.Cast(max_q_act_blob, to=caffe2_pb2.TensorProto.INT64)
        softmax_act_blob = C2.Cast(softmax_act_blob, to=caffe2_pb2.TensorProto.INT64)
        max_q_act_blob_nested, _ = C2.Reshape(max_q_act_blob, shape=[1, -1])
        softmax_act_blob_nested, _ = C2.Reshape(softmax_act_blob, shape=[1, -1])
        C2.net().Append(
            [max_q_act_blob_nested, softmax_act_blob_nested],
            [max_q_act_blob_nested]
        )
        transposed_action_idxs = C2.Transpose(max_q_act_blob_nested)
        flat_transposed_action_idxs = C2.FlattenToVec(transposed_action_idxs)
        output_values = 'output/int_single_categorical_features.values'
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.int64))
        C2.net().Copy([flat_transposed_action_idxs], [output_values])

        output_lengths = 'output/int_single_categorical_features.lengths'
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [flat_q_values_key], [output_lengths],
            value=2, dtype=caffe2_pb2.TensorProto.INT32)

        output_keys = 'output/int_single_categorical_features.keys'
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int64))
        output_keys_tensor, _ = C2.Concat(
            C2.ConstantFill(shape=[1, 1], value=0, dtype=caffe2_pb2.TensorProto.INT64),
            C2.ConstantFill(shape=[1, 1], value=1, dtype=caffe2_pb2.TensorProto.INT64),
            axis=0,
        )
        output_key_tile = C2.Tile(output_keys_tensor, num_examples, axis=0)
        C2.net().FlattenToVec([output_key_tile], [output_keys])

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        return ContinuousActionDQNPredictor(net, parameters)
