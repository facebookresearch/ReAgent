#!/usr/bin/env python3

import abc
from collections import OrderedDict
from io import BytesIO
from typing import List, Tuple

import torch
import torch.nn as nn


class ModelBase(nn.Module, metaclass=abc.ABCMeta):
    """
    A base class to support exporting through ONNX
    """

    @abc.abstractmethod
    def forward(self, input: OrderedDict) -> OrderedDict:
        """
        Args:
            input: An (nested) OrderedDict of torch.Tensor

        Returns:
            An OrderedDict of torch.Tensor
        """
        pass

    @abc.abstractmethod
    def input_prototype(self) -> OrderedDict:
        """
        This function provides the input for ONNX graph tracing.

        The return value should be what expected by `forward()`.
        """
        raise NotImplemented

    def export_to_buffer(self) -> BytesIO:
        """
        Export the instance to BytesIO buffer
        """
        export_model = ONNXExportModel(self)
        write_buffer = BytesIO()
        torch.onnx.export(
            export_model,
            export_model.onnx_input_args,
            write_buffer,
            input_names=export_model.onnx_input_names(),
        )
        return write_buffer


class ONNXExportModel(nn.Module):
    """
    This is a helper module to allow ONNX to trace instance of `ModelBase`.
    ONNX chooses to not trace dictionary input to module. To get around that,
    this module acts as an interface between ONNX and `ModelBase`. It gives
    flattened inputs to ONNX and structured input to `ModelBase`.
    """

    def __init__(self, m):
        super(ONNXExportModel, self).__init__()
        self.m = m
        self.input_prototype = m.input_prototype()
        self.onnx_input_args = self.flatten(self.input_prototype)

    def onnx_input_names(self) -> List[str]:
        """
        Returns names for ONNX to make the inputs more readable

        The names are the path to the tensor joint by ":".
        """

        def name_helper(d):
            if hasattr(d, "_asdict"):
                d = d._asdict()
            if isinstance(d, OrderedDict):
                return [
                    ":".join(filter(None, [k, n]))
                    for k, v in d.items()
                    for n in filter(lambda x: x is not None, name_helper(v))
                ]
            elif isinstance(d, torch.Tensor):
                return [""]
            elif d is None:
                return [None]
            else:
                raise ValueError()

        return list(filter(None, name_helper(self.input_prototype)))

    def structurize_input(self, args) -> OrderedDict:
        """
        Put args into input_prototype

        Since we expect the args to come from `self.onnx_input_args`, we can
        simply assert that the flatten forms are equivalent.
        """
        assert len(self.onnx_input_args) == len(args), "{} vs {}".format(
            len(self.onnx_input_args), len(args)
        )
        assert all(a is b for a, b in zip(self.onnx_input_args, args))
        return self.input_prototype

    def flatten(self, st) -> Tuple[torch.Tensor, ...]:
        """
        Flatten `st` to tuple of `torch.Tensor`s
        """

        def helper(st):
            if hasattr(st, "_asdict"):
                st = st._asdict()
            if isinstance(st, OrderedDict):
                return tuple(e for v in st.values() for e in helper(v))
            elif isinstance(st, torch.Tensor):
                return (st,)
            elif st is None:
                return (None,)
            else:
                raise ValueError()

        return tuple(filter(lambda x: x is not None, helper(st)))

    def forward(self, *args):
        structured_inputs = self.structurize_input(args)
        structured_outputs = self.m(structured_inputs)
        return self.flatten(structured_outputs)
