#!/usr/bin/env python3

import itertools
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

import caffe2.python.onnx.backend
import numpy as np
import onnx
import torch
import torch.nn as nn
from caffe2.python import workspace
from caffe2.python.core import BlobReference


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
    _model: Optional[Any] = None

    @staticmethod
    def set_net(net):
        C2._model = None
        C2._net = net

    @staticmethod
    def net():
        return C2._net

    @staticmethod
    def set_model(model):
        C2._model = model
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
    def pytorch_net_to_buffer(pytorch_net, input_dim, model_on_gpu):
        """Traces a pytorch net and outputs a python buffer object
        holding net."""

        if model_on_gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        write_buffer = BytesIO()
        dummy_input = torch.autograd.Variable(torch.randn(1, input_dim).type(dtype))
        torch.onnx._export(pytorch_net, dummy_input, write_buffer)
        return write_buffer

    @staticmethod
    def buffer_to_caffe2_netdef(buffer):
        """Creates caffe2 NetDef from buffer object and returns pointer to
        input and output blobs and the NetDef."""
        protobuf_model = onnx.load(BytesIO(buffer.getvalue()))
        input_blob_name = protobuf_model.graph.input[0].name
        outout_blob_name = protobuf_model.graph.output[0].name
        return (
            input_blob_name,
            outout_blob_name,
            caffe2.python.onnx.backend.prepare(protobuf_model),
        )


class WeightedEmbeddingBag(nn.EmbeddingBag):
    """A subclass of nn.EmbeddingBag that adds a constructor that supplies
    pre-existing embedding weights.
    """

    def __init__(
        self,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        mode="max",
        sparse=False,
    ):
        nn.Module.__init__(self)
        self.num_embeddings = weight.size(0)
        self.embedding_dim = weight.size(1)
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = weight
        self.mode = mode
        self.sparse = sparse
