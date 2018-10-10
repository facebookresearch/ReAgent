#!/usr/bin/env python3

import abc
from collections import OrderedDict
from io import BytesIO
from typing import List, NamedTuple, Tuple

import caffe2.python.onnx.backend
import onnx
import torch
import torch.nn as nn
from caffe2.python import core
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta


class ModelBase(nn.Module, metaclass=abc.ABCMeta):
    """
    A base class to support exporting through ONNX
    """

    @abc.abstractmethod
    def forward(self, input: NamedTuple) -> NamedTuple:
        """
        Args:
            input: An (nested) NamedTuple of torch.Tensor

        Returns:
            An NamedTuple of torch.Tensor
        """
        pass

    @abc.abstractmethod
    def input_prototype(self) -> NamedTuple:
        """
        This function provides the input for ONNX graph tracing.

        The return value should be what expected by `forward()`.
        """
        raise NotImplemented

    def input_blob_names(self) -> List[str]:
        """
        Returns the names of the input blobs. If `None`, the names will be
        derived from `self.input_prototype`.
        """
        return self.derived_input_blob_names()

    def derived_input_blob_names(self) -> List[str]:
        return self.derive_blob_names(self.input_prototype())

    def derive_blob_names(self, named_tuple):
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

        return list(filter(None, name_helper(named_tuple)))

    def output_blob_names(self) -> List[str]:
        """
        ONNX graph doesn't have good naming.

        Returns the names of the output blobs.
        """
        return self.derive_blob_names(self(self.input_prototype()))

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

    def get_caffe2_model(self):
        model_protobuf = onnx.load(BytesIO(self.export_to_buffer().getvalue()))
        input_blobs = [i.name for i in model_protobuf.graph.input]
        output_blobs = [o.name for o in model_protobuf.graph.output]
        return (
            caffe2.python.onnx.backend.prepare(model_protobuf),
            input_blobs,
            output_blobs,
        )

    def get_predictor_export_meta_and_workspace(self, feature_extractor=None):
        """

        ONNX would load blobs into a private workspace. We returns the workspace
        here instead of copying the blobs to the global workspace in order to
        save memory in the export state. Returning private workspace, we only
        need memory for PyTorch model, ONNX buffer, and Caffe2 model. Including
        optimizer parameters, this means we can train and save a model
        a quarter of the size of machine memory.

        We should revisit this once PyTorch 1.0 is ready.

        Args:
            feature_extractor: An instance of FeatureExtractorBase
        """
        # 1. Get Caffe2 model
        c2_model, input_blobs, output_blobs = self.get_caffe2_model()
        ws = c2_model.workspace

        # Initializing constants in the model
        init_net = core.Net(c2_model.init_net)
        ws.CreateNet(init_net)
        ws.RunNet(init_net)

        # Per ONNX code comment, input blobs are not initilized
        model_inputs = c2_model.uninitialized
        assert len(model_inputs) > 0, "Model is expected to have some input"
        parameters = [b for b in ws.Blobs() if b not in model_inputs]
        # Input blobs in order
        model_input_blobs = [b for b in input_blobs if b in model_inputs]

        predict_net = core.Net("predict_net")

        output_blob_names = self.output_blob_names()
        assert len(output_blobs) == len(output_blob_names), (
            "output_blobs and output_blob_names must have the same lengths. "
            "Check that your model don't reuse output tensors. "
            "output_blobs: {}; output_blob_names: {}".format(
                output_blobs, output_blob_names
            )
        )
        blob_remap = {
            onnx_name: explicit_name
            for onnx_name, explicit_name in zip(output_blobs, output_blob_names)
        }
        shapes = {}

        # 2. Create feature extractor net
        if feature_extractor:
            feature_extractor_nets = feature_extractor.create_net()
            # Initializing feature extractor parameters
            ws.CreateNet(feature_extractor_nets.init_net)
            ws.RunNet(feature_extractor_nets.init_net)
            feature_extractor_params = set(
                feature_extractor_nets.init_net.Proto().external_output
            )
            assert (
                len(set(parameters) & feature_extractor_params) == 0
            ), "Blob names collide! Please open a bug report"
            parameters += feature_extractor_params
            extracted_blobs = [
                str(b) for b in feature_extractor_nets.net.output_record().field_blobs()
            ]
            assert len(model_input_blobs) == len(extracted_blobs), (
                "The lengths of model_input_blobs and extracted_blobs must match. "
                "model_input_blobs: {}; extracted_blobs: {}".format(
                    model_input_blobs, extracted_blobs
                )
            )
            blob_remap.update(
                {
                    onnx_name: extracted_name
                    for onnx_name, extracted_name in zip(
                        model_input_blobs, extracted_blobs
                    )
                }
            )

            predict_net.AppendNet(feature_extractor_nets.net)
            del predict_net.Proto().external_output[:]

            input_blobs = [
                b
                for b in predict_net.Proto().external_input
                if b not in feature_extractor_params
            ]
            shapes.update({b: [] for b in input_blobs})
        else:
            input_blobs = model_input_blobs

        # 3. Rename the input blobs of model to match the output of feature
        #    extractor net

        model_net = core.Net(c2_model.predict_net).Clone(
            "remapped_model_net", blob_remap=blob_remap
        )

        # 5. Join feature extractor net & model net
        predict_net.AppendNet(model_net)

        # These shapes are not really used but required, so just pass fake ones
        shapes.update({b: [] for b in predict_net.Proto().external_output})

        return (
            PredictorExportMeta(
                predict_net,
                parameters,
                input_blobs,
                predict_net.Proto().external_output,
                shapes=shapes,
                net_type="async_scheduling",
                num_workers=8,
            ),
            ws,
        )


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

        explicit_names = self.m.input_blob_names()
        derived_names = self.m.derived_input_blob_names()

        if explicit_names is not None:
            assert len(explicit_names) == len(derived_names)
            return explicit_names

        return derived_names

    def structurize_input(self, args) -> NamedTuple:
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
