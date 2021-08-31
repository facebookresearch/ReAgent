#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging
from typing import Dict, Tuple

import reagent.core.types as rlt
import torch


logger = logging.getLogger(__name__)


@torch.jit.script
def map_id_list(raw_values: torch.Tensor, id2index: Dict[int, int]) -> torch.Tensor:
    # TODO(kaiwenw): handle case where raw_ids not in mapping
    # (i.e. id2index[val.item()] not found)
    return torch.tensor([id2index[x.item()] for x in raw_values], dtype=torch.long)


@torch.jit.script
def map_id_score_list(
    raw_keys: torch.Tensor, raw_values: torch.Tensor, id2index: Dict[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO(kaiwenw): handle case where raw_ids not in mapping
    # (i.e. id2index[val.item()] not found)
    return (
        torch.tensor([id2index[x.item()] for x in raw_keys], dtype=torch.long),
        raw_values,
    )


class MapIDList(torch.nn.Module):
    @abc.abstractmethod
    def forward(self, raw_values: torch.Tensor) -> torch.Tensor:
        pass


class MapIDScoreList(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self, raw_keys: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ExplicitMapIDList(MapIDList):
    def __init__(self, id2index: Dict[int, int]):
        super().__init__()
        self.id2index: Dict[int, int] = torch.jit.Attribute(id2index, Dict[int, int])

    def forward(self, raw_values: torch.Tensor) -> torch.Tensor:
        # TODO(kaiwenw): handle case where raw_ids not in mapping
        # (i.e. id2index[val.item()] not found)
        return torch.tensor(
            [self.id2index[x.item()] for x in raw_values], dtype=torch.long
        )


class ExplicitMapIDScoreList(MapIDScoreList):
    def __init__(self, id2index: Dict[int, int]):
        super().__init__()
        self.id2index: Dict[int, int] = torch.jit.Attribute(id2index, Dict[int, int])

    def forward(
        self, raw_keys: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO(kaiwenw): handle case where raw_ids not in mapping
        # (i.e. id2index[val.item()] not found)
        return (
            torch.tensor([self.id2index[x.item()] for x in raw_keys], dtype=torch.long),
            raw_values,
        )


class ModuloMapIDList(MapIDList):
    def __init__(self, modulo: int):
        super().__init__()
        self.modulo = modulo

    def forward(self, raw_values: torch.Tensor) -> torch.Tensor:
        return torch.remainder(raw_values.to(torch.long), self.modulo)


class ModuloMapIDScoreList(MapIDScoreList):
    def __init__(self, modulo: int):
        super().__init__()
        self.modulo = modulo

    def forward(
        self, raw_keys: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.remainder(raw_keys.to(torch.long), self.modulo),
            raw_values,
        )


def make_sparse_preprocessor(
    feature_config: rlt.ModelFeatureConfig, device: torch.device
):
    """Helper to initialize, for scripting SparsePreprocessor"""
    # TODO: Add option for simple modulo and other hash functions
    id2name: Dict[int, str] = feature_config.id2name
    name2id: Dict[str, int] = feature_config.name2id

    def _make_id_list_mapper(config: rlt.IdListFeatureConfig) -> MapIDList:
        mapping_config = feature_config.id_mapping_config[config.id_mapping_name].value
        if isinstance(mapping_config, rlt.ExplicitMapping):
            return ExplicitMapIDList(mapping_config.id2index)
        elif isinstance(mapping_config, rlt.ModuloMapping):
            return ModuloMapIDList(mapping_config.table_size)
        else:
            raise NotImplementedError(f"Unsupported {mapping_config}")

    id_list_mappers = {
        config.feature_id: _make_id_list_mapper(config)
        for config in feature_config.id_list_feature_configs
    }

    def _make_id_score_list_mapper(
        config: rlt.IdScoreListFeatureConfig,
    ) -> MapIDScoreList:
        mapping_config = feature_config.id_mapping_config[config.id_mapping_name].value
        if isinstance(mapping_config, rlt.ExplicitMapping):
            return ExplicitMapIDScoreList(mapping_config.id2index)
        elif isinstance(mapping_config, rlt.ModuloMapping):
            return ModuloMapIDScoreList(mapping_config.table_size)
        else:
            raise NotImplementedError(f"Unsupported {mapping_config}")

    id_score_list_mappers = {
        config.feature_id: _make_id_score_list_mapper(config)
        for config in feature_config.id_score_list_feature_configs
    }
    return torch.jit.script(
        SparsePreprocessor(
            id2name, name2id, id_list_mappers, id_score_list_mappers, device
        )
    )


class SparsePreprocessor(torch.nn.Module):
    """Performs preprocessing for sparse features (i.e. id_list, id_score_list)

    Functionality includes:
    (1) changes keys from feature_id to feature_name, for better debuggability
    (2) maps sparse ids to embedding table indices based on id_mapping
    (3) filters out ids which aren't in the id2name
    """

    def __init__(
        self,
        id2name: Dict[int, str],
        name2id: Dict[str, int],
        id_list_mappers: Dict[int, MapIDList],
        id_score_list_mappers: Dict[int, MapIDScoreList],
        device: torch.device,
    ) -> None:
        super().__init__()
        assert set(id2name.keys()) == set(id_list_mappers.keys()) | set(
            id_score_list_mappers.keys()
        )
        self.id2name: Dict[int, str] = torch.jit.Attribute(id2name, Dict[int, str])
        self.name2id: Dict[str, int] = torch.jit.Attribute(name2id, Dict[str, int])
        self.id_list_mappers = torch.nn.ModuleDict(
            {id2name[k]: v for k, v in id_list_mappers.items()}
        )
        self.id_score_list_mappers = torch.nn.ModuleDict(
            {id2name[k]: v for k, v in id_score_list_mappers.items()}
        )
        self.device = device

    @torch.jit.export
    def preprocess_id_list(
        self, id_list: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input: rlt.ServingIdListFeature
        Output: rlt.IdListFeature
        """
        ret: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, mapper in self.id_list_mappers.items():
            fid = self.name2id[name]
            if fid in id_list:
                offsets, values = id_list[fid]
                idx_values = mapper(values)
                ret[name] = (
                    offsets.to(self.device),
                    idx_values.to(self.device),
                )
        return ret

    @torch.jit.export
    def preprocess_id_score_list(
        self, id_score_list: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Input: rlt.ServingIdScoreListFeature
        Output: rlt.IdScoreListFeature
        """
        ret: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, mapper in self.id_score_list_mappers.items():
            fid = self.name2id[name]
            if fid in id_score_list:
                offsets, keys, values = id_score_list[fid]
                idx_keys, weights = mapper(keys, values)
                ret[name] = (
                    offsets.to(self.device),
                    idx_keys.to(self.device),
                    weights.to(self.device).float(),
                )
        return ret
