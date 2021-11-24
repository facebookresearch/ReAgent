#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from typing import Dict, List, Optional

import pytorch_lightning as pl
from reagent.core.parameters import NormalizationData


class ReAgentDataModule(pl.LightningDataModule):
    @abc.abstractmethod
    def get_normalization_data_map(
        self,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, NormalizationData]:
        pass
