#!/usr/bin/env python3

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta
from caffe2.python import model_helper, workspace
from ml.rl.caffe_utils import C2
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.training.rl_predictor import RLPredictor

import logging

logger = logging.getLogger(__name__)


class ContinuousActionDQNPredictor(RLPredictor):
    def __init__(self, net, parameters, int_features=False):
        RLPredictor.__init__(self, net, parameters, int_features)
        self.is_discrete = False
        self._output_blobs.extend(
            [
                "output/int_single_categorical_features.keys",
                "output/int_single_categorical_features.lengths",
                "output/int_single_categorical_features.values",
            ]
        )

    def predict(self, float_state_features, int_state_features, actions):
        """ Returns values for each state/action pair.

        :param float_state_features states as list of feature -> float value dict
        :param int_state_features states as list of feature -> int value dict
        :param actions actions as list of feature -> value dict
        """
        float_examples = []
        for i in range(len(float_state_features)):
            float_examples.append({**float_state_features[i], **actions[i]})
        if int_state_features is None:
            return RLPredictor.predict(self, float_examples)
        return RLPredictor.predict(self, float_examples, int_state_features)

    def policy(self, float_state_features, int_state_features, actions):
        float_examples = []
        for i in range(len(float_state_features)):
            float_examples.append({**float_state_features[i], **actions[i]})
        if int_state_features is None:
            return RLPredictor.policy(self, float_examples)
        return RLPredictor.policy(self, float_examples, int_state_features)

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
        int_features=False,
    ):
        """ Creates a ContinuousActionDQNPredictor from a ContinuousActionDQNTrainer.

        :param trainer ContinuousActionDQNTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param action_normalization_parameters action NormalizationParameters
        :param int_features boolean indicating if int features blob will be present
        """
        # ensure state and action IDs have no intersection
        assert (
            len(
                set(state_normalization_parameters.keys())
                & set(action_normalization_parameters.keys())
            )
            == 0
        )

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

        preprocessor = PreprocessorNet(True)
        parameters = []
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
        new_parameters, q_values = RLPredictor._forward_pass(
            model, trainer, state_action_normalized, ["Q"]
        )
        parameters.extend(new_parameters)

        flat_q_values_key = (
            "output/string_weighted_multi_categorical_features.values.values"
        )
        num_examples, _ = C2.Reshape(C2.Size(flat_q_values_key), shape=[1])
        q_value_blob, _ = C2.Reshape(flat_q_values_key, shape=[1, -1])

        # Get 1 x n (number of examples) action index tensor under the max_q policy
        max_q_act_idxs = "max_q_policy_actions"
        C2.net().FlattenToVec([C2.ArgMax(q_value_blob)], [max_q_act_idxs])
        max_q_act_blob = C2.Tile(max_q_act_idxs, num_examples, axis=0)

        # Get 1 x n (number of examples) action index tensor under the softmax policy
        temperature = C2.NextBlob("temperature")
        parameters.append(temperature)
        workspace.FeedBlob(
            temperature, np.array([trainer.rl_temperature], dtype=np.float32)
        )
        tempered_q_values = C2.Div(q_value_blob, temperature, broadcast=1)
        softmax_values = C2.Softmax(tempered_q_values)
        softmax_act_idxs_nested = "softmax_act_idxs_nested"
        C2.net().WeightedSample([softmax_values], [softmax_act_idxs_nested])
        softmax_act_blob = C2.Tile(
            C2.FlattenToVec(softmax_act_idxs_nested), num_examples, axis=0
        )

        # Concat action idx vecs to get 2 x n tensor [[a_maxq, ..], [a_softmax, ..]]
        # transpose & flatten to get [a_maxq, a_softmax, a_maxq, a_softmax, ...]
        max_q_act_blob = C2.Cast(max_q_act_blob, to=caffe2_pb2.TensorProto.INT64)
        softmax_act_blob = C2.Cast(softmax_act_blob, to=caffe2_pb2.TensorProto.INT64)
        max_q_act_blob_nested, _ = C2.Reshape(max_q_act_blob, shape=[1, -1])
        softmax_act_blob_nested, _ = C2.Reshape(softmax_act_blob, shape=[1, -1])
        C2.net().Append(
            [max_q_act_blob_nested, softmax_act_blob_nested], [max_q_act_blob_nested]
        )
        transposed_action_idxs = C2.Transpose(max_q_act_blob_nested)
        flat_transposed_action_idxs = C2.FlattenToVec(transposed_action_idxs)
        output_values = "output/int_single_categorical_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.int64))
        C2.net().Copy([flat_transposed_action_idxs], [output_values])

        output_lengths = "output/int_single_categorical_features.lengths"
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [flat_q_values_key],
            [output_lengths],
            value=2,
            dtype=caffe2_pb2.TensorProto.INT32,
        )

        output_keys = "output/int_single_categorical_features.keys"
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
        return ContinuousActionDQNPredictor(net, parameters, int_features)
