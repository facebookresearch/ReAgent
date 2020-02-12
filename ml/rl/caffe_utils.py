#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import logging
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from caffe2.python import core, workspace
from caffe2.python.core import BlobReference


logger = logging.getLogger(__name__)


class C2Meta(type):
    def __getattr__(cls, method_name):
        if method_name.startswith("__"):
            return super().__getattr__(method_name)

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
        retval: str = C2._net.NextBlob(prefix)
        return retval
