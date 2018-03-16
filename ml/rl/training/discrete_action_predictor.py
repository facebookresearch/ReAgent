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
from ml.rl.training.rl_predictor import RLPredictor
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet

import logging
logger = logging.getLogger(__name__)


class DiscreteActionPredictor(RLPredictor):
    def __init__(self, net, parameters):
        RLPredictor.__init__(self, net, parameters)
        self._output_blobs.extend([
            'output/string_single_categorical_features.keys',
            'output/string_single_categorical_features.lengths',
            'output/string_single_categorical_features.values',
        ])

    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def export(cls, trainer, actions, normalization_parameters):
        """ Creates DiscreteActionPredictor from a list of action trainers

        :param trainer DiscreteActionTrainer
        :param features list of state feature names
        :param actions list of action names
        """

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
        normalized_dense_matrix, new_parameters = preprocessor.normalize_sparse_matrix(
            'input/float_features.lengths',
            'input/float_features.keys',
            'input/float_features.values',
            normalization_parameters,
            'state_norm',
        )
        parameters.extend(new_parameters)

        new_parameters, q_values = RLPredictor._forward_pass(
            model,
            trainer,
            normalized_dense_matrix,
            actions,
        )
        parameters.extend(new_parameters)

        # Get 1 x n action index tensor under the max_q policy
        max_q_act_idxs = 'max_q_policy_actions'
        C2.net().Flatten([C2.RowWiseArgMax(q_values)], [max_q_act_idxs], axis=0)
        shape_of_num_of_states = 'num_states_shape'
        C2.net().FlattenToVec([max_q_act_idxs], [shape_of_num_of_states])
        num_states, _ = C2.Reshape(C2.Size(shape_of_num_of_states), shape=[1])

        # Get 1 x n action index tensor under the softmax policy
        temperature = C2.NextBlob("temperature")
        parameters.append(temperature)
        workspace.FeedBlob(
            temperature, np.array([trainer.rl_temperature], dtype=np.float32))
        tempered_q_values = C2.Div(q_values, "temperature", broadcast=1)
        softmax_values = C2.Softmax(tempered_q_values)
        tempered_softmax_values = C2.Div(softmax_values, "temperature", broadcast=1)
        softmax_act_idxs_nested = 'softmax_act_idxs_nested'
        C2.net().WeightedSample([tempered_softmax_values], [softmax_act_idxs_nested])
        softmax_act_idxs = 'softmax_policy_actions'
        C2.net().Flatten([softmax_act_idxs_nested], [softmax_act_idxs], axis=0)

        # Concat action index tensors to get 2 x n tensor - [[max_q], [softmax]]
        # transpose & flatten to get [a1_maxq, a1_softmax, a2_maxq, a2_softmax, ...]
        C2.net().Append([max_q_act_idxs, softmax_act_idxs], [max_q_act_idxs])
        transposed_action_idxs = C2.Transpose(max_q_act_idxs)
        flat_transposed_action_idxs = C2.FlattenToVec(transposed_action_idxs)
        output_values = 'output/string_single_categorical_features.values'
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.int64))
        C2.net().Gather(["action_names", flat_transposed_action_idxs], [output_values])

        output_lengths = 'output/string_single_categorical_features.lengths'
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [shape_of_num_of_states], [output_lengths],
            value=2, dtype=caffe2_pb2.TensorProto.INT32)

        output_keys = 'output/string_single_categorical_features.keys'
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int64))
        output_keys_tensor, _ = C2.Concat(
            C2.ConstantFill(shape=[1, 1], value=0, dtype=caffe2_pb2.TensorProto.INT64),
            C2.ConstantFill(shape=[1, 1], value=1, dtype=caffe2_pb2.TensorProto.INT64),
            axis=0,
        )
        output_key_tile = C2.Tile(output_keys_tensor, num_states, axis=0)
        C2.net().FlattenToVec([output_key_tile], [output_keys])

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        return DiscreteActionPredictor(net, parameters)
