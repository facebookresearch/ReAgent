#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from collections import defaultdict
from typing import List, Dict

import reagent.core.types as rlt
import torch
from torchrec import EmbeddingBagConfig

logger = logging.getLogger(__name__)


def embedding_bag_configs_from_feature_configs(
    configs: List[rlt.ModelFeatureConfig],
) -> List[EmbeddingBagConfig]:
    """
    Obtain a list of EmbeddingBagConfigs from multiple ModelFeatureConfigs.
    The returned list will be used for defining sparse model architectures
    """
    merged_id_mapping_config: Dict[str, rlt.IdMappingConfig] = {}
    for config in configs:
        for id_mapping_name, id_mapping_config in config.id_mapping_config.items():
            if id_mapping_name in merged_id_mapping_config:
                assert (
                    merged_id_mapping_config[id_mapping_name] == id_mapping_config
                ), f"Conflicting IdMappingConfigs for id_mapping_name={id_mapping_name}"
            else:
                merged_id_mapping_config[id_mapping_name] = id_mapping_config

    id_mapping_to_feature_names = defaultdict(list)
    for config in configs:
        for id_list_feature_config in config.id_list_feature_configs:
            id_mapping_to_feature_names[id_list_feature_config.id_mapping_name].append(
                id_list_feature_config.name
            )
        for id_score_list_feature_config in config.id_score_list_feature_configs:
            id_mapping_to_feature_names[
                id_score_list_feature_config.id_mapping_name
            ].append(id_score_list_feature_config.name)

    embedding_bag_configs: List[EmbeddingBagConfig] = []
    for id_mapping_name, config in merged_id_mapping_config.items():
        embedding_bag_configs.append(
            EmbeddingBagConfig(
                name=id_mapping_name,
                feature_names=id_mapping_to_feature_names[id_mapping_name],
                num_embeddings=config.embedding_table_size,
                embedding_dim=config.embedding_dim,
                pooling=config.pooling_type,
            )
        )
    logger.info(f"Generate EmbeddingBagConfigs: {embedding_bag_configs}")
    return embedding_bag_configs


def get_rank() -> int:
    """
    Returns the torch.distributed rank of the process. 0 represents
    the main process and is the default if torch.distributed isn't set up
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


class lazy_property(object):
    """
    More or less copy-pasta: http://stackoverflow.com/a/6849299
    Meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self._fget = fget
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, obj_cls_type):
        if obj is None:
            return None
        value = self._fget(obj)
        setattr(obj, self.__name__, value)
        return value
