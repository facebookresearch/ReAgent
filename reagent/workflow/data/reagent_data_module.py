#!/usr/bin/env python3

import abc
from typing import Dict, List

import pytorch_lightning as pl
from reagent.parameters import NormalizationData


class ReAgentDataModule(pl.LightningDataModule):
    @abc.abstractmethod
    def get_normalization_data_map(
        self, keys: List[str]
    ) -> Dict[str, NormalizationData]:
        pass
