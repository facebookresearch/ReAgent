#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging
from typing import cast, Dict, Tuple

import reagent.core.types as rlt
import torch


logger = logging.getLogger(__name__)


class MapIDList(torch.nn.Module):
    @abc.abstractmethod
    def forward(self, raw_ids: torch.Tensor) -> torch.Tensor:
        pass


class MapIDScoreList(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self, raw_ids: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ExactMapIDList(MapIDList):
    def __init__(self):
        super().__init__()

    def forward(self, raw_ids: torch.Tensor) -> torch.Tensor:
        return raw_ids


class ExactMapIDScoreList(MapIDScoreList):
    def __init__(self):
        super().__init__()

    def forward(
        self, raw_ids: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            raw_ids,
            raw_values,
        )


class HashingMapIDList(MapIDList):
    def __init__(self, embedding_table_size):
        super().__init__()
        self.embedding_table_size = embedding_table_size

    def forward(self, raw_ids: torch.Tensor) -> torch.Tensor:
        hashed_ids = torch.ops.fb.sigrid_hash(
            raw_ids,
            salt=0,
            maxValue=self.embedding_table_size,
            hashIntoInt32=False,
        )
        return hashed_ids


class HashingMapIDScoreList(MapIDScoreList):
    def __init__(self, embedding_table_size):
        super().__init__()
        self.embedding_table_size = embedding_table_size

    def forward(
        self, raw_ids: torch.Tensor, raw_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hashed_ids = torch.ops.fb.sigrid_hash(
            raw_ids,
            salt=0,
            maxValue=self.embedding_table_size,
            hashIntoInt32=False,
        )
        return (
            hashed_ids,
            raw_values,
        )


def make_sparse_preprocessor(
    feature_config: rlt.ModelFeatureConfig,
    device: torch.device,
    gen_torch_script: bool = True,
):
    """Helper to initialize, for scripting SparsePreprocessor"""
    # TODO: Add option for simple modulo and other hash functions
    id2name: Dict[int, str] = feature_config.id2name
    name2id: Dict[str, int] = feature_config.name2id

    def _make_id_list_mapper(config: rlt.IdListFeatureConfig) -> MapIDList:
        mapping_config = feature_config.id_mapping_config[config.id_mapping_name]
        if mapping_config.hashing:
            return HashingMapIDList(mapping_config.embedding_table_size)
        else:
            return ExactMapIDList()

    id_list_mappers = {
        config.feature_id: _make_id_list_mapper(config)
        for config in feature_config.id_list_feature_configs
    }

    def _make_id_score_list_mapper(
        config: rlt.IdScoreListFeatureConfig,
    ) -> MapIDScoreList:
        mapping_config = feature_config.id_mapping_config[config.id_mapping_name]
        if mapping_config.hashing:
            return HashingMapIDScoreList(mapping_config.embedding_table_size)
        else:
            return ExactMapIDScoreList()

    id_score_list_mappers = {
        config.feature_id: _make_id_score_list_mapper(config)
        for config in feature_config.id_score_list_feature_configs
    }
    sparse_preprocessor = SparsePreprocessor(
        id2name,
        name2id,
        id_list_mappers,
        id_score_list_mappers,
        device,
        gen_torch_script,
    )
    if gen_torch_script:
        return torch.jit.script(sparse_preprocessor)
    else:
        return sparse_preprocessor


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
        gen_torch_script: bool,
    ) -> None:
        super().__init__()
        assert set(id2name.keys()) == set(id_list_mappers.keys()) | set(
            id_score_list_mappers.keys()
        )
        self._id2name: Dict[int, str] = id2name
        self._name2id: Dict[str, int] = name2id
        if gen_torch_script:
            self.id2name: Dict[int, str] = torch.jit.Attribute(id2name, Dict[int, str])
            self.name2id: Dict[str, int] = torch.jit.Attribute(name2id, Dict[str, int])
        else:
            self.id2name: Dict[int, str] = id2name
            self.name2id: Dict[str, int] = name2id
        self._id_list_mappers = id_list_mappers
        self._id_score_list_mappers = id_score_list_mappers
        self.id_list_mappers = torch.nn.ModuleDict(
            {id2name[k]: v for k, v in id_list_mappers.items()}
        )
        self.id_score_list_mappers = torch.nn.ModuleDict(
            {id2name[k]: v for k, v in id_score_list_mappers.items()}
        )

        self.device = device
        self.gen_torch_script = gen_torch_script

    @torch.jit.ignore
    def __getstate__(self):
        # Return a dictionary of picklable attributes
        # Notice, this method should not be scripted!
        # This method is for pickle only, which is used in Inter Process Communication (IPC)
        # , i.e., copy objects across multi-processes
        return {
            "id2name": self._id2name,
            "name2id": self._name2id,
            "id_list_mappers": self._id_list_mappers,
            "id_score_list_mappers": self._id_score_list_mappers,
            "device": self.device,
            "gen_torch_script": self.gen_torch_script,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        # Set object attributes from pickled state dictionary
        # Notice, this method should not be scripted!
        # This method is for pickle only, which is used in Inter Process Communication (IPC)
        # , i.e., copy objects across multi-processes

        # Notice, we need to re-init the module so that the attributes can be assigned
        super().__init__()
        self.gen_torch_script = state["gen_torch_script"]
        if self.gen_torch_script:
            raise AssertionError(
                "SparsePreprocessor should not be scripted at this stage!"
            )
        self.id2name = cast(Dict[int, str], state["id2name"])
        self.name2id = cast(Dict[int, str], state["name2id"])
        self._id_list_mappers = state["id_list_mappers"]
        self._id_score_list_mappers = state["id_score_list_mappers"]
        self.device = state["device"]

        self.id_list_mappers = torch.nn.ModuleDict(
            {self.id2name[k]: v for k, v in self._id_list_mappers.items()}
        )
        self.id_score_list_mappers = torch.nn.ModuleDict(
            {self.id2name[k]: v for k, v in self._id_score_list_mappers.items()}
        )

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
