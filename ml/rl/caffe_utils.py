#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import logging
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

import caffe2.python.onnx.backend
import numpy as np
import onnx
import torch
from caffe2.python import core, workspace
from caffe2.python.core import BlobReference


logger = logging.getLogger(__name__)


class C2Meta(type):
    def __getattr__(cls, method_name):
        def method(*inputs, **kwargs):
            tb = traceback.extract_stack(limit=2)
            blob_prefix = "{}:{}:{}".format(
                os.path.basename(tb[0].filename), tb[0].lineno, method_name
            )
            OpSchema = workspace.C.OpSchema
            schema = OpSchema.get(method_name)
            num_outputs = schema.CalculateOutput(len(inputs))
            outputs = []
            if num_outputs < 0:
                num_outputs = schema.max_output
            for x in range(num_outputs):
                outputs.append(C2._net.NextBlob(blob_prefix + "_output" + str(x)))

            promoted_inputs = []
            for i in inputs:
                if type(i) != str and type(i) != BlobReference:
                    # Promote input by stuffing into a blob
                    input_name = C2._net.NextBlob(blob_prefix + "_input" + str(x))
                    if type(i) == np.ndarray:
                        workspace.FeedBlob(input_name, i)
                    else:
                        workspace.FeedBlob(input_name, np.array([i], dtype=np.float32))
                    promoted_inputs.append(input_name)
                else:
                    promoted_inputs.append(i)
            return C2._net.__getattr__(method_name)(promoted_inputs, outputs, **kwargs)

        return method


class C2(metaclass=C2Meta):
    _net: Optional[Any] = None
    _init_net: Optional[Any] = None
    _model: Optional[Any] = None

    @staticmethod
    def set_net(net):
        C2._model = None
        C2._net = net
        C2._init_net = None

    @staticmethod
    def set_net_and_init_net(net, init_net):
        C2._model = None
        C2._net = net
        C2._init_net = init_net

    @staticmethod
    def net():
        return C2._net

    @staticmethod
    def init_net():
        return C2._init_net

    @staticmethod
    def set_model(model):
        C2._model = model
        C2._init_net = None
        if model is None:
            C2._net = None
        else:
            C2._net = model.net

    @staticmethod
    def model():
        return C2._model

    @staticmethod
    def NextBlob(prefix: str) -> str:
        assert C2._net is not None
        tb = traceback.extract_stack(limit=2)
        prefix = "{}:{}:{}:{}".format(
            C2._net.Name(), os.path.basename(tb[0].filename), tb[0].lineno, prefix
        )
        return C2._net.NextBlob(prefix)


class StackedArray(object):
    def __init__(self, lengths, values):
        self.lengths = lengths
        self.values = values

    @classmethod
    def from_list_list(cls, d: List[List[float]], blob_prefix: str):
        lengths_blob = blob_prefix + "_lengths"
        values_blob = blob_prefix + "_values"

        workspace.FeedBlob(lengths_blob, np.array([len(x) for x in d], dtype=np.int32))

        workspace.FeedBlob(
            values_blob, np.array(list(itertools.chain(*d)), dtype=np.float32)
        )

        return cls(lengths_blob, values_blob)


class StackedAssociativeArray(object):
    def __init__(self, lengths, keys, values):
        self.lengths = lengths
        self.keys = keys
        self.values = values

    def to_python(self) -> List[Dict[Any, Any]]:
        keys = workspace.FetchBlob(self.keys)
        lengths = workspace.FetchBlob(self.lengths)
        values = workspace.FetchBlob(self.values)
        retval: List[Dict[Any, Any]] = []
        cursor = 0
        for length in lengths:
            d = {}
            for _ in range(length):
                key = keys[cursor]
                value = values[cursor]
                d[key] = value
                cursor += 1
            retval.append(d)
        return retval

    @classmethod
    def from_dict_list(cls, d: List[Dict[int, float]], blob_prefix: str):
        lengths_blob = blob_prefix + "_lengths"
        keys_blob = blob_prefix + "_keys"
        values_blob = blob_prefix + "_values"

        workspace.FeedBlob(lengths_blob, np.array([len(x) for x in d], dtype=np.int32))

        key_list_2d = [list(x.keys()) for x in d]
        workspace.FeedBlob(
            keys_blob, np.array(list(itertools.chain(*key_list_2d)), dtype=np.int32)
        )

        value_list_2d = [list(x.values()) for x in d]
        workspace.FeedBlob(
            values_blob,
            np.array(list(itertools.chain(*value_list_2d)), dtype=np.float32),
        )

        return cls(lengths_blob, keys_blob, values_blob)


class StackedTwoLevelAssociativeArray(object):
    def __init__(
        self,
        outer_lengths: str,
        outer_keys: str,
        inner_lengths: str,
        inner_keys: str,
        inner_values: str,
    ) -> None:
        self.outer_lengths = outer_lengths
        self.outer_keys = outer_keys
        self.inner_lengths = inner_lengths
        self.inner_keys = inner_keys
        self.inner_values = inner_values

    def to_python(self) -> List[Dict[Any, Dict[Any, Any]]]:
        outer_keys = workspace.FetchBlob(self.outer_keys)
        outer_lengths = workspace.FetchBlob(self.outer_lengths)
        inner_keys = workspace.FetchBlob(self.inner_keys)
        inner_lengths = workspace.FetchBlob(self.inner_lengths)
        inner_values = workspace.FetchBlob(self.inner_values)
        retval: List[Dict[Any, Dict[Any, Any]]] = []
        outer_cursor = 0
        inner_cursor = 0
        for length in outer_lengths:
            outer_dict = {}
            for _ in range(length):
                outer_key = outer_keys[outer_cursor]
                inner_length = inner_lengths[outer_cursor]
                outer_cursor += 1
                inner_dict = {}
                for _ in range(inner_length):
                    inner_key = inner_keys[inner_cursor]
                    inner_value = inner_values[inner_cursor]
                    inner_cursor += 1
                    inner_dict[inner_key] = inner_value
                outer_dict[outer_key] = inner_dict
            retval.append(outer_dict)
        return retval


class PytorchCaffe2Converter(object):
    @staticmethod
    def pytorch_net_to_caffe2_netdef(*args, **kwargs):
        buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(*args, **kwargs)
        return PytorchCaffe2Converter.buffer_to_caffe2_netdef(buffer)

    @staticmethod
    def pytorch_net_to_buffer(pytorch_net, input_dim, model_on_gpu, float_input=True):
        """Traces a pytorch net and outputs a python buffer object
        holding net."""

        training = pytorch_net.training
        pytorch_net.train(False)

        for name, p in pytorch_net.named_parameters():
            inf_count = torch.isinf(p).sum().item()
            nan_count = torch.isnan(p).sum().item()
            assert inf_count + nan_count == 0, "{} has {} inf and {} nan".format(
                name, inf_count, nan_count
            )

        if float_input:
            dtype = torch.cuda.FloatTensor if model_on_gpu else torch.FloatTensor
            dummy_input = torch.randn(1, input_dim).type(dtype)
        else:
            dtype = torch.cuda.LongTensor if model_on_gpu else torch.LongTensor
            dummy_input = torch.randint(low=0, high=1, size=(1, input_dim)).type(dtype)

        write_buffer = BytesIO()
        try:
            torch.onnx.export(pytorch_net, dummy_input, write_buffer)
        finally:
            pytorch_net.train(training)
        return write_buffer

    @staticmethod
    def buffer_to_caffe2_netdef(buffer):
        """Creates caffe2 NetDef from buffer object and returns pointer to
        input and output blobs and the NetDef."""
        protobuf_model = onnx.load(BytesIO(buffer.getvalue()))
        input_blob_name = protobuf_model.graph.input[0].name
        output_blob_name = protobuf_model.graph.output[0].name
        logger.info(
            "INPUT BLOB: " + input_blob_name + ". OUTPUT BLOB:" + output_blob_name
        )
        return (
            input_blob_name,
            output_blob_name,
            caffe2.python.onnx.backend.prepare(protobuf_model),
        )

    @staticmethod
    def remap_blobs(input_blob, output_blob, netdef, prefix):
        init_net = core.Net(netdef.init_net)
        predict_net = core.Net(netdef.predict_net)

        blob_remap = {
            str(b): "{}/{}".format(prefix, str(b))
            for n in [init_net, predict_net]
            for b in n.external_inputs + n.external_outputs
        }

        remapped_input_blob = blob_remap[input_blob]
        remapped_output_blob = blob_remap[output_blob]

        remapped_init_net, _blob_remap = core.clone_and_bind_net(
            init_net, "{}_init".format(prefix), "{}_init/".format(prefix), blob_remap
        )
        remapped_predict_net, predict_blob_remap = core.clone_and_bind_net(
            predict_net,
            "{}_predict".format(prefix),
            "{}_predict/".format(prefix),
            blob_remap,
        )

        torch_workspace = netdef.workspace

        parameters = torch_workspace.Blobs()
        for blob_str in parameters:
            workspace.FeedBlob(
                blob_remap[blob_str], torch_workspace.FetchBlob(blob_str)
            )

        remapped_parameters = [predict_blob_remap[b] for b in parameters]
        return (
            remapped_input_blob,
            remapped_output_blob,
            remapped_parameters,
            remapped_init_net,
            remapped_predict_net,
        )


def softmax(x, temperature):
    """Compute softmax values for each sets of scores in x."""
    x = x / temperature
    return torch.nn.functional.softmax(x, dim=1)


def masked_softmax(x, mask, temperature):
    """Compute softmax values for each sets of scores in x."""
    x = x / temperature
    mask_min_x = x - ((1.0 - mask) * 1e20)
    mask_min_x -= torch.max(mask_min_x, dim=1, keepdim=True)[0]
    e_x = torch.exp(mask_min_x)
    e_x *= mask
    out = e_x / e_x.sum(dim=1, keepdim=True)

    # Set NaN values to 0 (NaN happens when a full mask row is passed in)
    out[out != out] = 0
    return out
