#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import reagent.types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    InputColumn,
    PolicyNetworkBatchPreprocessor,
    Preprocessor,
)
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorUnwrapper as ActorPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import (  # type: ignore
        ActorPredictorUnwrapper,
    )


logger = logging.getLogger(__name__)


def get_feature_config(
    float_features: Optional[List[Tuple[int, str]]]
) -> rlt.ModelFeatureConfig:
    float_features = float_features or []
    float_feature_infos = [
        rlt.FloatFeatureInfo(name=f_name, feature_id=f_id)
        for f_id, f_name in float_features
    ]

    return rlt.ModelFeatureConfig(float_feature_infos=float_feature_infos)


@dataclass
class ActorCriticBase(ModelManager):
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    action_preprocessing_options: Optional[PreprocessingOptions] = None
    action_feature_override: Optional[str] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    action_float_features: List[Tuple[int, str]] = field(default_factory=list)
    reader_options: Optional[ReaderOptions] = None

    def __post_init_post_parse__(self):
        super().__init__()
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.whitelist_features is None
        ), (
            "Please set state whitelist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.action_preprocessing_options is None
            or self.action_preprocessing_options.whitelist_features is None
        ), (
            "Please set action whitelist features in action_float_features field of "
            "config instead"
        )
        self._state_preprocessing_options = self.state_preprocessing_options
        self._action_preprocessing_options = self.action_preprocessing_options
        self._action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None
        self._actor_network: Optional[ModelBase] = None
        self._q1_network: Optional[ModelBase] = None
        self._metrics_to_score = None

    def create_policy(self, serving: bool) -> Policy:
        """ Create online actor critic policy. """

        from reagent.gym.policies.samplers.continuous_sampler import GaussianSampler
        from reagent.gym.policies.scorers.continuous_scorer import sac_scorer
        from reagent.gym.policies import ActorPredictorPolicy

        if serving:
            return ActorPredictorPolicy(
                ActorPredictorUnwrapper(self.build_serving_module())
            )
        else:
            sampler = GaussianSampler(self.trainer.actor_network)
            scorer = sac_scorer(self.trainer.actor_network)
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def metrics_to_score(self) -> List[str]:
        assert self._reward_options is not None
        if self._metrics_to_score is None:
            self._metrics_to_score = get_metrics_to_score(
                self._reward_options.metric_reward_values
            )
        return self._metrics_to_score

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        assert len(self.action_float_features) > 0, "You must set action_float_features"
        return get_feature_config(self.action_float_features)

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        print("State_feature_config: ", self.state_feature_config)
        print("Action_feature_config: ", self.action_feature_config)

        # Run state feature identification
        state_preprocessing_options = (
            self._state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state whitelist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            whitelist_features=state_features
        )

        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec, InputColumn.STATE_FEATURES, state_preprocessing_options
        )

        # Run action feature identification
        action_preprocessing_options = (
            self._action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id for ffi in self.action_feature_config.float_feature_infos
        ]
        logger.info(f"action whitelist_features: {action_features}")

        actor_net_builder = self.actor_net_builder.value
        action_feature_override = actor_net_builder.default_action_preprocessing
        logger.info(f"Default action_feature_override is {action_feature_override}")
        if self.action_feature_override is not None:
            action_feature_override = self.action_feature_override

        assert action_preprocessing_options.feature_overrides is None
        action_preprocessing_options = action_preprocessing_options._replace(
            whitelist_features=action_features,
            feature_overrides={fid: action_feature_override for fid in action_features},
        )
        action_normalization_parameters = identify_normalization_parameters(
            input_table_spec, InputColumn.ACTION, action_preprocessing_options
        )

        return {
            InputColumn.STATE_FEATURES: NormalizationData(
                dense_normalization_parameters=state_normalization_parameters
            ),
            InputColumn.ACTION: NormalizationData(
                dense_normalization_parameters=action_normalization_parameters
            ),
        }

    @property
    def state_normalization_parameters(self) -> Dict[int, NormalizationParameters]:
        return self.get_float_features_normalization_parameters(
            InputColumn.STATE_FEATURES
        )

    @property
    def action_normalization_parameters(self) -> Dict[int, NormalizationParameters]:
        return self.get_float_features_normalization_parameters(InputColumn.ACTION)

    def _set_normalization_parameters(
        self, normalization_data_map: Dict[str, NormalizationData]
    ):
        """
        Set normalization parameters on current instance
        """
        state_norm_data = normalization_data_map.get(InputColumn.STATE_FEATURES, None)
        assert state_norm_data is not None
        assert state_norm_data.dense_normalization_parameters is not None
        action_norm_data = normalization_data_map.get(InputColumn.ACTION, None)
        assert action_norm_data is not None
        assert action_norm_data.dense_normalization_parameters is not None
        self.set_normalization_data_map(normalization_data_map)

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError()

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        return PolicyNetworkBatchPreprocessor(
            state_preprocessor=Preprocessor(
                self.state_normalization_parameters, use_gpu=self.use_gpu
            ),
            action_preprocessor=Preprocessor(
                self.action_normalization_parameters, use_gpu=self.use_gpu
            ),
            use_gpu=self.use_gpu,
        )

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
    ) -> RLTrainingOutput:
        """
        Train the model
        """
        raise NotImplementedError()
