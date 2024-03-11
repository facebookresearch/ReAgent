#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from copy import deepcopy
from typing import Any, Optional

import torch.nn as nn
from reagent.core import types as rlt


# add ABCMeta once https://github.com/sphinx-doc/sphinx/issues/5995 is fixed
class ModelBase(nn.Module):
    """
    A base class to support exporting through ONNX
    """

    def input_prototype(self) -> Any:
        """
        This function provides the input for ONNX graph tracing.

        The return value should be what expected by `forward()`.
        """
        raise NotImplementedError

    def feature_config(self) -> Optional[rlt.ModelFeatureConfig]:
        """
        If the model needs additional preprocessing, e.g., using sequence features,
        returns the config here.
        """
        return None

    def get_target_network(self) -> "ModelBase":
        """
        Return a copy of this network to be used as target network

        Subclass should override this if the target network should share parameters
        with the network to be trained.
        """
        return deepcopy(self)

    def get_distributed_data_parallel_model(self):
        """
        Return DistributedDataParallel version of this model

        This needs to be implemented explicitly because:
        1) Model with EmbeddingBag module is not compatible with vanilla DistributedDataParallel
        2) Exporting logic needs structured data. DistributedDataParallel doesn't work with structured data.
        """
        raise NotImplementedError

    def cpu_model(self) -> "ModelBase":
        """
        Override this in DistributedDataParallel models
        """
        # This is not ideal but makes exporting simple
        return deepcopy(self).cpu()

    def requires_model_parallel(self) -> bool:
        """Return True if this model has large embedding tables which need to be sharded"""
        return False
