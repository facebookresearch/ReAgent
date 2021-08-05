#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, NormalizationKey
from reagent.data import DataFetcher, ReAgentDataModule
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.top_k_sampler import TopKSampler
from reagent.gym.policies.scorers.slate_q_scorer import slate_q_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.preprocessing.normalization import get_feature_config
from reagent.reporting.slate_q_reporter import SlateQReporter
from reagent.training import ReAgentLightningModule
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)


logger = logging.getLogger(__name__)


@dataclass
class SlateQBase(ModelManager):
    slate_feature_id: int = 0
    slate_score_id: Tuple[int, int] = (0, 0)
    item_preprocessing_options: Optional[PreprocessingOptions] = None
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    item_float_features: Optional[List[Tuple[int, str]]] = None

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.allowedlist_features is None
        ), (
            "Please set state allowlist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.allowedlist_features is None
        ), (
            "Please set item allowlist features in item_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.sequence_feature_id is None
        ), "Please set slate_feature_id field of config instead"
        self._state_preprocessing_options = self.state_preprocessing_options
        self._item_preprocessing_options = self.item_preprocessing_options
        self.eval_parameters = self.trainer_param.evaluation

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ):
        if serving:
            assert normalization_data_map
            return create_predictor_policy_from_model(
                self.build_serving_module(trainer_module, normalization_data_map),
                # pyre-fixme[16]: `SlateQBase` has no attribute `num_candidates`.
                max_num_actions=self.num_candidates,
                # pyre-fixme[16]: `SlateQBase` has no attribute `slate_size`.
                slate_size=self.slate_size,
            )
        else:
            scorer = slate_q_scorer(
                num_candidates=self.num_candidates,
                # pyre-fixme[6]: Expected `ModelBase` for 2nd param but got
                #  `Union[torch.Tensor, torch.nn.Module]`.
                q_network=trainer_module.q_network,
            )
            sampler = TopKSampler(k=self.slate_size)
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def item_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.item_float_features)

    def get_reporter(self):
        return SlateQReporter()
