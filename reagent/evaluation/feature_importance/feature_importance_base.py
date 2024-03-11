#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
from typing import List

import pandas as pd
import torch.nn as nn
from reagent.core.dataclasses import dataclass


@dataclass
class FeatureImportanceBase:
    model: nn.Module
    sorted_feature_ids: List[int]

    def compute_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError()
