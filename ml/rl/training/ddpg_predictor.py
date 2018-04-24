#!/usr/bin/env python3

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, workspace

from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet


class DDPGPredictor(object):
    def __init__(self, net, trainer) -> None:
        self._net = net
        self.trainer = trainer

    @classmethod
    def export(cls, trainer, state_normalization_parameters):
        """Export caffe2 preprocessor net and pytorch actor forward pass as one
        caffe2 net.
        """
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(trainer.actor)
        actor_input_blob, actor_output_blob, caffe2_netdef =\
            PytorchCaffe2Converter.buffer_to_caffe2_netdef(buffer)
        torch_workspace = caffe2_netdef.workspace

        for blob in torch_workspace.Blobs():
            workspace.FeedBlob(blob, torch_workspace.FetchBlob(blob))

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
        net.Copy([state_normalized_dense_matrix], ['states'])
        parameters.extend(new_parameters)
        workspace.RunNetOnce(model.param_init_net)

        net.Copy([state_normalized_dense_matrix], [actor_input_blob])
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
        workspace.FeedBlob(output_keys, np.zeros(1, dtype=np.float32))
        C2.net().FlattenToVec([actor_output_blob], [output_values])

        workspace.CreateNet(net)
        return DDPGPredictor(net, trainer)
