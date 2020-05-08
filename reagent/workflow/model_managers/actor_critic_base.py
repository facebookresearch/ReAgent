#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import Evaluator, get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.parameters import (
    NormalizationData,
    NormalizationKey,
    NormalizationParameters,
)
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    InputColumn,
    PolicyNetworkBatchPreprocessor,
    Preprocessor,
)
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.reporters.actor_critic_reporter import ActorCriticReporter
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    RLTrainingReport,
    TableSpec,
)
from reagent.workflow.utils import train_and_evaluate_generic
from reagent.workflow_utils.page_handler import (
    EvaluationPageHandler,
    TrainingPageHandler,
)


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorUnwrapper as ActorPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import ActorPredictorUnwrapper


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


class ActorPolicyWrapper(Policy):
    """ Actor's forward function is our act """

    def __init__(self, actor_network):
        self.actor_network = actor_network

    @torch.no_grad()
    # pyre-fixme[14]: `act` overrides method defined in `Policy` inconsistently.
    # pyre-fixme[14]: `act` overrides method defined in `Policy` inconsistently.
    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        self.actor_network.eval()
        output = self.actor_network(obs)
        self.actor_network.train()
        return output.detach().cpu()


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
        self._metrics_to_score = None

    @property
    def should_generate_eval_dataset(self) -> bool:
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `eval_parameters`.
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `eval_parameters`.
        return self.eval_parameters.calc_cpe_in_training

    def create_policy(self, serving: bool) -> Policy:
        """ Create online actor critic policy. """

        from reagent.gym.policies import ActorPredictorPolicy

        if serving:
            return ActorPredictorPolicy(
                ActorPredictorUnwrapper(self.build_serving_module())
            )
        else:
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `_actor_network`.
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `_actor_network`.
            return ActorPolicyWrapper(self._actor_network)

    @property
    def metrics_to_score(self) -> List[str]:
        assert self._reward_options is not None
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `_metrics_to_score`.
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `_metrics_to_score`.
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
        # Run state feature identification
        state_preprocessing_options = (
            # pyre-fixme[16]: `ActorCriticBase` has no attribute
            #  `_state_preprocessing_options`.
            # pyre-fixme[16]: `ActorCriticBase` has no attribute
            #  `_state_preprocessing_options`.
            self._state_preprocessing_options
            or PreprocessingOptions()
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
            # pyre-fixme[16]: `ActorCriticBase` has no attribute
            #  `_action_preprocessing_options`.
            # pyre-fixme[16]: `ActorCriticBase` has no attribute
            #  `_action_preprocessing_options`.
            self._action_preprocessing_options
            or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id for ffi in self.action_feature_config.float_feature_infos
        ]
        logger.info(f"action whitelist_features: {action_features}")

        # pyre-fixme[16]: `ActorCriticBase` has no attribute `actor_net_builder`.
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `actor_net_builder`.
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
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=state_normalization_parameters
            ),
            NormalizationKey.ACTION: NormalizationData(
                dense_normalization_parameters=action_normalization_parameters
            ),
        }

    @property
    def state_normalization_parameters(self) -> Dict[int, NormalizationParameters]:
        return self.get_float_features_normalization_parameters(NormalizationKey.STATE)

    @property
    def action_normalization_parameters(self) -> Dict[int, NormalizationParameters]:
        return self.get_float_features_normalization_parameters(NormalizationKey.ACTION)

    def _set_normalization_parameters(
        self, normalization_data_map: Dict[str, NormalizationData]
    ):
        """
        Set normalization parameters on current instance
        """
        state_norm_data = normalization_data_map.get(NormalizationKey.STATE, None)
        assert state_norm_data is not None
        assert state_norm_data.dense_normalization_parameters is not None
        action_norm_data = normalization_data_map.get(NormalizationKey.ACTION, None)
        assert action_norm_data is not None
        assert action_norm_data.dense_normalization_parameters is not None
        self.set_normalization_data_map(normalization_data_map)

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        logger.info("Starting query")
        return query_data(
            input_table_spec=input_table_spec,
            discrete_action=False,
            include_possible_actions=False,
            custom_reward_expression=reward_options.custom_reward_expression,
            sample_range=sample_range,
        )

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

    # TODO: deprecate, once we deprecate internal page handlers
    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
    ) -> RLTrainingOutput:

        reporter = ActorCriticReporter()
        # pyre-fixme[16]: `RLTrainer` has no attribute `add_observer`.
        self.trainer.add_observer(reporter)

        evaluator = Evaluator(
            action_names=None,
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `rl_parameters`.
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `rl_parameters`.
            gamma=self.rl_parameters.gamma,
            model=self.trainer,
            metrics_to_score=self.metrics_to_score,
        )
        # pyre-fixme[16]: `Evaluator` has no attribute `add_observer`.
        # pyre-fixme[16]: `Evaluator` has no attribute `add_observer`.
        evaluator.add_observer(reporter)

        batch_preprocessor = self.build_batch_preprocessor()
        train_and_evaluate_generic(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            trainer=self.trainer,
            num_epochs=num_epochs,
            use_gpu=self.use_gpu,
            batch_preprocessor=batch_preprocessor,
            reporter=reporter,
            evaluator=evaluator,
            reader_options=self.reader_options,
        )
        # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
        # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
        training_report = RLTrainingReport.make_union_instance(
            reporter.generate_training_report()
        )

        # TODO: save actor/critic weights
        return RLTrainingOutput(training_report=training_report)
