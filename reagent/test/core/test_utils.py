#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import unittest

import reagent.core.types as rlt
from reagent.core.utils import embedding_bag_configs_from_feature_configs


class TestUtils(unittest.TestCase):
    def test_embedding_bag_configs_from_feature_configs(self) -> None:
        TABLE_1_EMBED_SIZE = 100
        TABLE_1_EMBED_DIM = 64
        TABLE_2_EMBED_SIZE = 200
        TABLE_2_EMBED_DIM = 32

        feature_config_1 = rlt.ModelFeatureConfig(
            float_feature_infos=[rlt.FloatFeatureInfo(name="dummy0", feature_id=0)],
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name="id_list_feature_111",
                    feature_id=111,
                    id_mapping_name="table_1",
                )
            ],
            id_score_list_feature_configs=[
                rlt.IdScoreListFeatureConfig(
                    name="id_score_list_feature_112",
                    feature_id=112,
                    id_mapping_name="table_2",
                )
            ],
            id_mapping_config={
                "table_1": rlt.IdMappingConfig(
                    embedding_table_size=TABLE_1_EMBED_SIZE,
                    embedding_dim=TABLE_1_EMBED_DIM,
                ),
                "table_2": rlt.IdMappingConfig(
                    embedding_table_size=TABLE_2_EMBED_SIZE,
                    embedding_dim=TABLE_2_EMBED_DIM,
                ),
            },
        )
        feature_config_2 = rlt.ModelFeatureConfig(
            float_feature_infos=[rlt.FloatFeatureInfo(name="dummy1", feature_id=1)],
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name="id_list_feature_211",
                    feature_id=211,
                    id_mapping_name="table_1",
                )
            ],
            id_score_list_feature_configs=[
                rlt.IdScoreListFeatureConfig(
                    name="id_score_list_feature_212",
                    feature_id=212,
                    id_mapping_name="table_1",
                )
            ],
            id_mapping_config={
                "table_1": rlt.IdMappingConfig(
                    embedding_table_size=TABLE_1_EMBED_SIZE,
                    embedding_dim=TABLE_1_EMBED_DIM,
                ),
            },
        )
        embedding_bag_configs = embedding_bag_configs_from_feature_configs(
            [feature_config_1, feature_config_2]
        )
        assert len(embedding_bag_configs) == 2

        assert embedding_bag_configs[0].name == "table_1"
        assert embedding_bag_configs[0].num_embeddings == TABLE_1_EMBED_SIZE
        assert embedding_bag_configs[0].embedding_dim == TABLE_1_EMBED_DIM
        assert embedding_bag_configs[0].feature_names == [
            "id_list_feature_111",
            "id_list_feature_211",
            "id_score_list_feature_212",
        ]

        assert embedding_bag_configs[1].name == "table_2"
        assert embedding_bag_configs[1].num_embeddings == TABLE_2_EMBED_SIZE
        assert embedding_bag_configs[1].embedding_dim == TABLE_2_EMBED_DIM
        assert embedding_bag_configs[1].feature_names == ["id_score_list_feature_112"]

        # feature_config_3 specifies inconsistent id_mapping_config as those in feature_config_1
        # we expect to see exception
        feature_config_3 = rlt.ModelFeatureConfig(
            float_feature_infos=[rlt.FloatFeatureInfo(name="dummy1", feature_id=1)],
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name="id_list_feature_211",
                    feature_id=211,
                    id_mapping_name="table_1",
                )
            ],
            id_score_list_feature_configs=[
                rlt.IdScoreListFeatureConfig(
                    name="id_score_list_feature_212",
                    feature_id=212,
                    id_mapping_name="table_1",
                )
            ],
            id_mapping_config={
                "table_1": rlt.IdMappingConfig(
                    embedding_table_size=TABLE_1_EMBED_SIZE + 1,
                    embedding_dim=TABLE_1_EMBED_DIM + 1,
                ),
            },
        )
        self.assertRaises(
            AssertionError,
            embedding_bag_configs_from_feature_configs,
            [feature_config_1, feature_config_3],
        )
