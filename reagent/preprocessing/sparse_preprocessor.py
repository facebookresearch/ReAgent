#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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


def make_sparse_preprocessor(
    feature_config: rlt.ModelFeatureConfig, device: torch.device, jit_scripted=True
):
    """Helper to initialize, for scripting SparsePreprocessor"""
    id2name: Dict[int, str] = feature_config.id2name
    id2mapping: Dict[int, Dict[int, int]] = {
        fid: feature_config.id_mapping_config[
            feature_config.id2config[fid].id_mapping_name
        ].id2index
        for fid in feature_config.id2config
    }
    sparse_preprocessor = SparsePreprocessor(id2name, id2mapping, device)
    if jit_scripted:
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
        id2mapping: Dict[int, Dict[int, int]],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.id2name: Dict[int, str] = torch.jit.Attribute(id2name, Dict[int, str])
        self.id2mapping: Dict[int, Dict[int, int]] = torch.jit.Attribute(
            id2mapping, Dict[int, Dict[int, int]]
        )
        assert set(id2name.keys()) == set(id2mapping.keys())
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
        for fid, (offsets, values) in id_list.items():
            if fid in self.id2name:
                id2index = self.id2mapping[fid]
                idx_values = map_id_list(values, id2index)
                ret[self.id2name[fid]] = (
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
        for fid, (offsets, keys, values) in id_score_list.items():
            if fid in self.id2name:
                id2index = self.id2mapping[fid]
                idx_keys, weights = map_id_score_list(keys, values, id2index)
                ret[self.id2name[fid]] = (
                    offsets.to(self.device),
                    idx_keys.to(self.device),
                    weights.to(self.device),
                )
        return ret
