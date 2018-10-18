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


class DDPGPredictor(RLPredictor):
    def __init__(self, net, init_net, parameters, int_features) -> None:
        RLPredictor.__init__(self, net, init_net, parameters, int_features)
        self._output_blobs = [
            "output/float_features.lengths",
            "output/float_features.keys",
            "output/float_features.values",
        ]

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

    @classmethod
    def export_actor(
        cls,
        trainer,
        state_normalization_parameters,
        min_action_range_tensor_serving,
        max_action_range_tensor_serving,
        int_features=False,
        model_on_gpu=False,
    ):
        """Export caffe2 preprocessor net and pytorch actor forward pass as one
        caffe2 net.

        :param trainer DDPGTrainer
        :param state_normalization_parameters state NormalizationParameters
        :param min_action_range_tensor_serving pytorch tensor that specifies
            min action value for each dimension
        :param max_action_range_tensor_serving pytorch tensor that specifies
            min action value for each dimension
        :param state_normalization_parameters state NormalizationParameters
        :param int_features boolean indicating if int features blob will be present
        :param model_on_gpu boolean indicating if the model is a GPU model or CPU model
        """
        input_dim = trainer.state_dim
        if isinstance(trainer.actor, DataParallel):
            trainer.actor = trainer.actor.module

        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.actor, input_dim, model_on_gpu
        )
        actor_input_blob, actor_output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
            buffer
        )
        torch_workspace = caffe2_netdef.workspace

        parameters = torch_workspace.Blobs()
        for blob_str in parameters:
            workspace.FeedBlob(blob_str, torch_workspace.FetchBlob(blob_str))

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)
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

        # Feed action scaling tensors for serving
        min_action_serving_blob = C2.NextBlob("min_action_range_tensor_serving")
        workspace.FeedBlob(
            min_action_serving_blob, min_action_range_tensor_serving.cpu().data.numpy()
        )
        parameters.append(str(min_action_serving_blob))

        max_action_serving_blob = C2.NextBlob("max_action_range_tensor_serving")
        workspace.FeedBlob(
            max_action_serving_blob, max_action_range_tensor_serving.cpu().data.numpy()
        )
        parameters.append(str(max_action_serving_blob))

        # Feed action scaling tensors for training [-1, 1] due to tanh actor
        min_vals_training = trainer.min_action_range_tensor_training.cpu().data.numpy()
        min_action_training_blob = C2.NextBlob("min_action_range_tensor_training")
        workspace.FeedBlob(min_action_training_blob, min_vals_training)
        parameters.append(str(min_action_training_blob))

        max_vals_training = trainer.max_action_range_tensor_training.cpu().data.numpy()
        max_action_training_blob = C2.NextBlob("max_action_range_tensor_training")
        workspace.FeedBlob(max_action_training_blob, max_vals_training)
        parameters.append(str(max_action_training_blob))

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
        sorted_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        state_dense_matrix, new_parameters = sparse_to_dense(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            sorted_features,
        )
        parameters.extend(new_parameters)
        state_normalized_dense_matrix, new_parameters = preprocessor.normalize_dense_matrix(
            state_dense_matrix,
            sorted_features,
            state_normalization_parameters,
            "state_norm",
            False,
        )
        parameters.extend(new_parameters)

        net.Copy([state_normalized_dense_matrix], [actor_input_blob])

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(torch_init_net)

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

        output_keys_int32 = "output_keys_int32"
        output_keys = "output/float_features.keys"
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int64))
        C2.net().LengthsRangeFill([output_lengths], [output_keys_int32])
        C2.net().Cast(
            [output_keys_int32], [output_keys], to=caffe2_pb2.TensorProto.INT64
        )

        output_values = "output/float_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        # Scale actors actions from [-1, 1] to serving range
        prev_range = C2.Sub(max_action_training_blob, min_action_training_blob)
        new_range = C2.Sub(max_action_serving_blob, min_action_serving_blob)
        subtract_prev_min = C2.Sub(actor_output_blob, min_action_training_blob)
        div_by_prev_range = C2.Div(subtract_prev_min, prev_range)
        scaled_for_serving_actions = C2.Add(
            C2.Mul(div_by_prev_range, new_range), min_action_serving_blob
        )
        C2.net().FlattenToVec([scaled_for_serving_actions], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, torch_init_net, parameters, int_features)

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
        if isinstance(trainer.critic, DataParallel):
            trainer.critic = trainer.critic.module

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

        preprocessor = PreprocessorNet(True)
        sorted_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        state_dense_matrix, new_parameters = sparse_to_dense(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            sorted_features,
        )
        parameters.extend(new_parameters)
        state_normalized_dense_matrix, new_parameters = preprocessor.normalize_dense_matrix(
            state_dense_matrix,
            sorted_features,
            state_normalization_parameters,
            "state_norm",
            False,
        )
        parameters.extend(new_parameters)

        # Don't normalize actions, just go from sparse -> dense
        action_dense_matrix, new_parameters = sparse_to_dense(
            input_feature_lengths,
            input_feature_keys,
            input_feature_values,
            list(action_normalization_parameters.keys()),  # TODO: Clean up in D10161240
        )
        parameters.extend(new_parameters)
        state_action_normalized = "state_action_normalized"
        state_action_normalized_dim = "state_action_normalized_dim"
        net.Concat(
            [state_normalized_dense_matrix, action_dense_matrix],
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

        output_keys_int32 = "output_keys_int32"
        output_keys = "output/float_features.keys"
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.int64))
        C2.net().LengthsRangeFill([output_lengths], [output_keys_int32])
        C2.net().Cast(
            [output_keys_int32], [output_keys], to=caffe2_pb2.TensorProto.INT64
        )

        output_values = "output/float_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([critic_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, torch_init_net, parameters, int_features)
