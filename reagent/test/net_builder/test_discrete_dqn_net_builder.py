#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from typing import Optional

from reagent import types as rlt
from reagent.net_builder import discrete_dqn
from reagent.net_builder.unions import DiscreteDQNNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.preprocessing.identify_types import CONTINUOUS


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorWrapper as DiscreteDqnPredictorWrapper,
        FbDiscreteDqnPredictorWrapperWithIdList as DiscreteDqnPredictorWrapperWithIdList,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import (
        DiscreteDqnPredictorWrapper,
        DiscreteDqnPredictorWrapperWithIdList,
    )


class TestDiscreteDQNNetBuilder(unittest.TestCase):
    def _test_discrete_dqn_net_builder(
        self,
        chooser: DiscreteDQNNetBuilder__Union,
        state_feature_config: Optional[rlt.ModelFeatureConfig] = None,
        serving_module_class=DiscreteDqnPredictorWrapper,
    ) -> None:
        builder = chooser.value
        state_dim = 3
        state_feature_config = state_feature_config or rlt.ModelFeatureConfig(
            float_feature_infos=[
                rlt.FloatFeatureInfo(name=f"f{i}", feature_id=i)
                for i in range(state_dim)
            ]
        )
        state_dim = len(state_feature_config.float_feature_infos)

        state_normalization_data = NormalizationData(
            dense_normalization_parameters={
                fi.feature_id: NormalizationParameters(
                    feature_type=CONTINUOUS, mean=0.0, stddev=1.0
                )
                for fi in state_feature_config.float_feature_infos
            }
        )

        action_names = ["L", "R"]
        q_network = builder.build_q_network(
            state_feature_config, state_normalization_data, len(action_names)
        )
        x = q_network.input_prototype()
        y = q_network(x)
        self.assertEqual(y.shape, (1, 2))
        serving_module = builder.build_serving_module(
            q_network, state_normalization_data, action_names, state_feature_config
        )
        self.assertIsInstance(serving_module, serving_module_class)

    def test_fully_connected(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = DiscreteDQNNetBuilder__Union(
            FullyConnected=discrete_dqn.fully_connected.FullyConnected()
        )
        self._test_discrete_dqn_net_builder(chooser)

    def test_dueling(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = DiscreteDQNNetBuilder__Union(Dueling=discrete_dqn.dueling.Dueling())
        self._test_discrete_dqn_net_builder(chooser)

    def test_fully_connected_with_id_list_none(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = DiscreteDQNNetBuilder__Union(
            FullyConnectedWithEmbedding=discrete_dqn.fully_connected_with_embedding.FullyConnectedWithEmbedding()
        )
        self._test_discrete_dqn_net_builder(
            chooser, serving_module_class=DiscreteDqnPredictorWrapperWithIdList
        )

    def test_fully_connected_with_id_list(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = DiscreteDQNNetBuilder__Union(
            FullyConnectedWithEmbedding=discrete_dqn.fully_connected_with_embedding.FullyConnectedWithEmbedding()
        )
        state_feature_config = rlt.ModelFeatureConfig(
            float_feature_infos=[
                rlt.FloatFeatureInfo(name=str(i), feature_id=i) for i in range(1, 5)
            ],
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name="A", feature_id=10, id_mapping_name="A_mapping"
                )
            ],
            id_mapping_config={"A_mapping": rlt.IdMapping(ids=[0, 1, 2])},
        )
        self._test_discrete_dqn_net_builder(
            chooser,
            state_feature_config=state_feature_config,
            serving_module_class=DiscreteDqnPredictorWrapperWithIdList,
        )
