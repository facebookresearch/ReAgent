#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import abc
from typing import Dict, List, Optional

import pytorch_lightning as pl
from reagent.core.parameters import NormalizationData


class ReAgentDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def get_normalization_data_map(
        self,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, NormalizationData]:
        pass

    @abc.abstractproperty
    def train_dataset(self):
        pass

    @abc.abstractproperty
    def eval_dataset(self):
        pass

    @abc.abstractproperty
    def test_dataset(self):
        pass
