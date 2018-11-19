#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, workspace
from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.preprocessing.sparse_to_dense import sparse_to_dense
from ml.rl.training.rl_predictor_pytorch import RLPredictor
from torch.nn import DataParallel


logger = logging.getLogger(__name__)

OUTPUT_SINGLE_CAT_KEYS_NAME = "output/string_single_categorical_features.keys"
OUTPUT_SINGLE_CAT_LENGTHS_NAME = "output/string_single_categorical_features.lengths"
OUTPUT_SINGLE_CAT_VALS_NAME = "output/string_single_categorical_features.values"


class DQNPredictor(RLPredictor):
    def __init__(self, net, init_net, parameters, int_features):
        RLPredictor.__init__(self, net, init_net, parameters, int_features)
        self._output_blobs.extend(
            [
                OUTPUT_SINGLE_CAT_KEYS_NAME,
                OUTPUT_SINGLE_CAT_LENGTHS_NAME,
                OUTPUT_SINGLE_CAT_VALS_NAME,
            ]
        )

    @classmethod
    def export(
        cls,
        trainer,
        actions,
        state_normalization_parameters,
        int_features=False,
        model_on_gpu=False,
    ):
        """Export caffe2 preprocessor net and pytorch DQN forward pass as one
        caffe2 net.

        :param trainer DQNTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param int_features boolean indicating if int features blob will be present
        :param model_on_gpu boolean indicating if the model is a GPU model or CPU model
        """

        input_dim = trainer.num_features

        if isinstance(trainer.q_network, DataParallel):
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

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)
        logger.info("Generated ONNX predict net:")
        logger.info(str(torch_predict_net.Proto()))
        # While converting to metanetdef, the external_input of predict_net
        # will be recomputed. Add the real output of init_net to parameters
        # to make sure they will be counted.
        parameters.extend(
            set(caffe2_netdef.init_net.external_output)
            - set(caffe2_netdef.init_net.external_input)
        )

        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        C2.set_model(model)

        workspace.FeedBlob("input/image", np.zeros([1, 1, 1, 1], dtype=np.int32))
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

        if state_normalization_parameters is not None:
            sorted_feature_ids = sort_features_by_normalization(
                state_normalization_parameters
            )[0]
            dense_matrix, new_parameters = sparse_to_dense(
                input_feature_lengths,
                input_feature_keys,
                input_feature_values,
                sorted_feature_ids,
            )
            parameters.extend(new_parameters)
            preprocessor_net = PreprocessorNet()
            state_normalized_dense_matrix, new_parameters = preprocessor_net.normalize_dense_matrix(
                dense_matrix,
                sorted_feature_ids,
                state_normalization_parameters,
                "state_norm_",
                True,
            )
            parameters.extend(new_parameters)
        else:
            # Image input.  Note: Currently this does the wrong thing if
            #   more than one image is passed at a time.
            state_normalized_dense_matrix = "input/image"

        net.Copy([state_normalized_dense_matrix], [qnet_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

        net.AppendNet(torch_predict_net)

        new_parameters, q_values = RLPredictor._forward_pass(
            model, trainer, state_normalized_dense_matrix, actions, qnet_output_blob
        )
        parameters.extend(new_parameters)

        # Get 1 x n action index tensor under the max_q policy
        max_q_act_idxs = "max_q_policy_actions"
        C2.net().Flatten([C2.ArgMax(q_values)], [max_q_act_idxs], axis=0)
        shape_of_num_of_states = "num_states_shape"
        C2.net().FlattenToVec([max_q_act_idxs], [shape_of_num_of_states])
        num_states, _ = C2.Reshape(C2.Size(shape_of_num_of_states), shape=[1])

        # Get 1 x n action index tensor under the softmax policy
        temperature = C2.NextBlob("temperature")
        parameters.append(temperature)
        workspace.FeedBlob(
            temperature, np.array([trainer.rl_temperature], dtype=np.float32)
        )
        tempered_q_values = C2.Div(q_values, temperature, broadcast=1)
        softmax_values = C2.Softmax(tempered_q_values)
        softmax_act_idxs_nested = "softmax_act_idxs_nested"
        C2.net().WeightedSample([softmax_values], [softmax_act_idxs_nested])
        softmax_act_idxs = "softmax_policy_actions"
        C2.net().Flatten([softmax_act_idxs_nested], [softmax_act_idxs], axis=0)

        action_names = C2.NextBlob("action_names")
        parameters.append(action_names)
        workspace.FeedBlob(action_names, np.array(actions))

        # Concat action index tensors to get 2 x n tensor - [[max_q], [softmax]]
        # transpose & flatten to get [a1_maxq, a1_softmax, a2_maxq, a2_softmax, ...]
        max_q_act_blob = C2.Cast(max_q_act_idxs, to=caffe2_pb2.TensorProto.INT32)
        softmax_act_blob = C2.Cast(softmax_act_idxs, to=caffe2_pb2.TensorProto.INT32)
        C2.net().Append([max_q_act_blob, softmax_act_blob], [max_q_act_blob])
        transposed_action_idxs = C2.Transpose(max_q_act_blob)
        flat_transposed_action_idxs = C2.FlattenToVec(transposed_action_idxs)
        workspace.FeedBlob(OUTPUT_SINGLE_CAT_VALS_NAME, np.zeros(1, dtype=np.int64))
        C2.net().Gather(
            [action_names, flat_transposed_action_idxs], [OUTPUT_SINGLE_CAT_VALS_NAME]
        )

        workspace.FeedBlob(OUTPUT_SINGLE_CAT_LENGTHS_NAME, np.zeros(1, dtype=np.int32))
        C2.net().ConstantFill(
            [shape_of_num_of_states],
            [OUTPUT_SINGLE_CAT_LENGTHS_NAME],
            value=2,
            dtype=caffe2_pb2.TensorProto.INT32,
        )

        workspace.FeedBlob(OUTPUT_SINGLE_CAT_KEYS_NAME, np.zeros(1, dtype=np.int64))
        output_keys_tensor, _ = C2.Concat(
            C2.ConstantFill(shape=[1, 1], value=0, dtype=caffe2_pb2.TensorProto.INT64),
            C2.ConstantFill(shape=[1, 1], value=1, dtype=caffe2_pb2.TensorProto.INT64),
            axis=0,
        )
        output_key_tile = C2.Tile(output_keys_tensor, num_states, axis=0)
        C2.net().FlattenToVec([output_key_tile], [OUTPUT_SINGLE_CAT_KEYS_NAME])

        workspace.CreateNet(net)
        return DQNPredictor(net, torch_init_net, parameters, int_features)
