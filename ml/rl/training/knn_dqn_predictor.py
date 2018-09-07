#!/usr/bin/env python3

import logging

import faiss
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, workspace
from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.training.rl_predictor_pytorch import RLPredictor


logger = logging.getLogger(__name__)


class KNNDQNPredictor(RLPredictor):
    def __init__(
        self,
        net,
        init_net,
        parameters,
        int_features=False,
        actor_predictor=None,
        critic_predictor=None,
        action_embedding=None,
        action_names=None,
        k=1,
    ) -> None:
        RLPredictor.__init__(self, net, init_net, parameters, int_features)
        self.actor_predictor = actor_predictor
        self.critic_predictor = critic_predictor
        if action_embedding is not None:
            assert len(action_embedding.shape) == 2
            self.index = faiss.IndexFlatIP(action_embedding.shape[1])
            self.index.add(action_embedding)

        self.action_embedding = action_embedding
        self.action_names = action_names
        self._output_blobs = [
            "output/float_features.lengths",
            "output/float_features.keys",
            "output/float_features.values",
        ]
        self.k = k

    def repeat(self, examples):
        if examples is None:
            return None
        return [ex for ex in examples for _i in range(self.k)]

    def predict(self, float_states, int_state_features=None):
        proto_actions = self.actor_predictor.actor_prediction(
            float_states, int_state_features
        )
        _scores, actions = self.index.search(proto_actions, self.k)
        actions = actions.reshape(-1)
        q_values = self.critic_predictor.critic_prediction(
            self.repeat(float_states), self.repeat(int_state_features), actions
        )

        def get_action_name(a):
            return a if self.action_names is None else self.action_names[a]

        action_values = list(zip(actions, q_values))
        return [
            {
                get_action_name(a): q
                for a, q in action_values[i * self.k : (i + 1) * self.k]
            }
            for i in range(len(float_states))
        ]

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

        lengths = workspace.FetchBlob("output/float_features.lengths")
        results = workspace.FetchBlob("output/float_features.values").reshape(
            len(lengths), -1
        )
        return results

    def critic_prediction(self, float_state_features, int_state_features, actions):
        """ Critic Prediction - Returns values for each state/action pair. Accepts
        int_features as 3rd optional parameter.

        :param float_state_features states as list of feature -> float value dict
        :param int_state_features states as list of feature -> int value dict
        :param actions actions as list of feature -> value dict
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
        workspace.FeedBlob("input/actions", actions)

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
    def export(
        cls,
        trainer,
        state_normalization_parameters,
        int_features=False,
        model_on_gpu=False,
        action_names=None,
    ):
        actor_predictor = cls.export_actor(
            trainer, state_normalization_parameters, int_features, model_on_gpu
        )
        critic_predictor = cls.export_critic(
            trainer, state_normalization_parameters, int_features, model_on_gpu
        )
        action_embedding = trainer.action_embedding.weight.detach().numpy()
        return cls(
            net=None,
            init_net=None,
            parameters=None,
            int_features=int_features,
            actor_predictor=actor_predictor,
            critic_predictor=critic_predictor,
            action_embedding=action_embedding,
            action_names=action_names,
            k=trainer.k,
        )

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
        :param model_on_gpu boolean indicating if the model is a GPU model or CPU model
        """
        input_dim = trainer.state_dim
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
            trainer.actor, input_dim, model_on_gpu
        )
        actor_input_blob, actor_output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
            buffer
        )

        torch_init_net = core.Net(caffe2_netdef.init_net)
        torch_predict_net = core.Net(caffe2_netdef.predict_net)

        blob_remap = {
            str(b): "actor/" + str(b)
            for n in [torch_init_net, torch_predict_net]
            for b in n.external_inputs + n.external_outputs
        }

        actor_input_blob = blob_remap[actor_input_blob]
        actor_output_blob = blob_remap[actor_output_blob]

        torch_init_net, _blob_remap = core.clone_and_bind_net(
            torch_init_net, "torch_actor_init", "torch_actor_init/", blob_remap
        )
        torch_predict_net, _blob_remap = core.clone_and_bind_net(
            torch_predict_net, "torch_actor_predict", "torch_actor_predict/", blob_remap
        )

        torch_workspace = caffe2_netdef.workspace

        parameters = torch_workspace.Blobs()
        for blob_str in parameters:
            workspace.FeedBlob(
                blob_remap[blob_str], torch_workspace.FetchBlob(blob_str)
            )

        model = model_helper.ModelHelper(name="actor")
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
        return KNNDQNPredictor(net, torch_init_net, parameters, int_features)

    @classmethod
    def export_critic(
        cls,
        trainer,
        state_normalization_parameters,
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
        embedding_input_blob, embedding_output_blob, embedding_caffe2_netdef = PytorchCaffe2Converter.pytorch_net_to_caffe2_netdef(
            trainer.action_embedding, 1, model_on_gpu, float_input=False
        )
        critic_input_blob, critic_output_blob, caffe2_netdef = PytorchCaffe2Converter.pytorch_net_to_caffe2_netdef(
            trainer.critic, input_dim, model_on_gpu
        )

        embedding_input_blob, embedding_output_blob, embedding_parameters, embedding_init_net, embedding_predict_net = PytorchCaffe2Converter.remap_blobs(
            embedding_input_blob,
            embedding_output_blob,
            embedding_caffe2_netdef,
            "embedding",
        )
        critic_input_blob, critic_output_blob, critic_parameters, critic_init_net, critic_predict_net = PytorchCaffe2Converter.remap_blobs(
            critic_input_blob, critic_output_blob, caffe2_netdef, "critic"
        )

        parameters = embedding_parameters + critic_parameters

        model = model_helper.ModelHelper(name="predictor")
        net = model.net
        C2.set_model(model)

        workspace.FeedBlob("input/float_features.lengths", np.zeros(1, dtype=np.int32))
        workspace.FeedBlob("input/float_features.keys", np.zeros(1, dtype=np.int64))
        workspace.FeedBlob("input/float_features.values", np.zeros(1, dtype=np.float32))

        workspace.FeedBlob("input/actions", np.zeros(1, dtype=np.int64))

        C2.net().Copy(["input/actions"], embedding_input_blob)

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

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(embedding_init_net)
        workspace.RunNetOnce(critic_init_net)

        net.AppendNet(embedding_init_net)
        net.AppendNet(embedding_predict_net)

        state_action_normalized = "state_action_normalized"
        state_action_normalized_dim = "state_action_normalized_dim"
        net.Concat(
            [state_normalized_dense_matrix, embedding_output_blob],
            [state_action_normalized, state_action_normalized_dim],
            axis=1,
        )
        net.Copy([state_action_normalized], [critic_input_blob])

        net.AppendNet(critic_init_net)
        net.AppendNet(critic_predict_net)

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
        C2.net().Copy([embedding_input_blob], [output_keys])

        output_values = "output/float_features.values"
        workspace.FeedBlob(output_values, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([critic_output_blob], [output_values])

        workspace.CreateNet(net)
        return KNNDQNPredictor(net, critic_init_net, parameters, int_features)
