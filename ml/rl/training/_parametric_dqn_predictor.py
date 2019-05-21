#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, workspace
from ml.rl.caffe_utils import C2, PytorchCaffe2Converter, StackedAssociativeArray
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.preprocessing.sparse_to_dense import Caffe2SparseToDenseProcessor
from ml.rl.training.rl_predictor_pytorch import RLPredictor
from torch.nn.parallel.distributed import DistributedDataParallel


logger = logging.getLogger(__name__)

OUTPUT_SINGLE_CAT_KEYS_NAME = "output/int_single_categorical_features.keys"
OUTPUT_SINGLE_CAT_LENGTHS_NAME = "output/int_single_categorical_features.lengths"
OUTPUT_SINGLE_CAT_VALS_NAME = "output/int_single_categorical_features.values"


class _ParametricDQNPredictor(RLPredictor):
    def __init__(self, net, init_net, parameters):
        RLPredictor.__init__(self, net, init_net, parameters)
        self._output_blobs.extend(
            [
                OUTPUT_SINGLE_CAT_KEYS_NAME,
                OUTPUT_SINGLE_CAT_LENGTHS_NAME,
                OUTPUT_SINGLE_CAT_VALS_NAME,
            ]
        )

    def predict(self, float_state_features, actions):
        """ Returns values for each state/action pair.

        :param float_state_features states as list of feature -> float value dict
        :param actions actions as list of feature -> value dict
        """
        float_examples = []
        for i in range(len(float_state_features)):
            float_examples.append({**float_state_features[i], **actions[i]})
        return RLPredictor.predict(self, float_examples)

    @classmethod
    def export(
        cls,
        trainer,
        state_normalization_parameters,
        action_normalization_parameters,
        model_on_gpu=False,
        normalize_actions=True,
    ):
        """Export caffe2 preprocessor net and pytorch DQN forward pass as one
        caffe2 net.

        :param trainer ParametricDQNTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param action_normalization_parameters action NormalizationParameters
        :param model_on_gpu boolean indicating if the model is a GPU model or CPU model
        """

        input_dim = trainer.num_features
        if isinstance(trainer.q_network, DistributedDataParallel):
            trainer.q_network = trainer.q_network.module

        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.q_network, input_dim, model_on_gpu
        )
        qnet_input_blob, qnet_output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
            buffer
        )
        torch_workspace = caffe2_netdef.workspace

        parameters = torch_workspace.Blobs()
        for blob_str in parameters:
            workspace.FeedBlob(blob_str, torch_workspace.FetchBlob(blob_str))

        # Remove the input blob from parameters since it's not a real
        #     input (will be calculated by preprocessor)
        parameters.remove(qnet_input_blob)

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)
        # While converting to metanetdef, the external_input of predict_net
        # will be recomputed. Add the real output of init_net to parameters
        # to make sure they will be counted.
        parameters.extend(
            set(caffe2_netdef.init_net.external_output)
            - set(caffe2_netdef.init_net.external_input)
        )

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

        C2.net().Copy(["input/float_features.lengths"], [input_feature_lengths])
        C2.net().Copy(["input/float_features.keys"], [input_feature_keys])
        C2.net().Copy(["input/float_features.values"], [input_feature_values])

        preprocessor = PreprocessorNet()
        sparse_to_dense_processor = Caffe2SparseToDenseProcessor()
        sorted_state_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        state_dense_matrix, new_parameters = sparse_to_dense_processor(
            sorted_state_features,
            StackedAssociativeArray(
                input_feature_lengths, input_feature_keys, input_feature_values
            ),
        )
        parameters.extend(new_parameters)
        state_normalized_dense_matrix, new_parameters = preprocessor.normalize_dense_matrix(
            state_dense_matrix,
            sorted_state_features,
            state_normalization_parameters,
            "state_norm",
            False,
        )
        parameters.extend(new_parameters)

        sorted_action_features, _ = sort_features_by_normalization(
            action_normalization_parameters
        )
        action_dense_matrix, new_parameters = sparse_to_dense_processor(
            sorted_action_features,
            StackedAssociativeArray(
                input_feature_lengths, input_feature_keys, input_feature_values
            ),
        )
        parameters.extend(new_parameters)
        if normalize_actions:
            action_normalized_dense_matrix, new_parameters = preprocessor.normalize_dense_matrix(
                action_dense_matrix,
                sorted_action_features,
                action_normalization_parameters,
                "action_norm",
                False,
            )
            parameters.extend(new_parameters)
        else:
            action_normalized_dense_matrix = action_dense_matrix

        state_action_normalized = "state_action_normalized"
        state_action_normalized_dim = "state_action_normalized_dim"
        net.Concat(
            [state_normalized_dense_matrix, action_normalized_dense_matrix],
            [state_action_normalized, state_action_normalized_dim],
            axis=1,
        )

        net.Copy([state_action_normalized], [qnet_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_predict_net)

        new_parameters, q_values = RLPredictor._forward_pass(
            model, trainer, state_action_normalized, ["Q"], qnet_output_blob
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

        workspace.CreateNet(net)
        return _ParametricDQNPredictor(net, torch_init_net, parameters)
