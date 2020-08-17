#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import (
    Dataset,
    ModelFeatureConfigProvider__Union,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.evaluation.evaluator import Evaluator, get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.discrete_scorer import discrete_dqn_scorer
from reagent.models.base import ModelBase
from reagent.models.model_feature_config_provider import RawModelFeatureConfigProvider
from reagent.parameters import EvaluationParameters, NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    DiscreteDqnBatchPreprocessor,
)
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.preprocessing.types import InputColumn
from reagent.reporting.discrete_dqn_reporter import DiscreteDQNReporter
from reagent.workflow.model_managers.model_manager import ModelManager


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQNBase(ModelManager):
    target_action_distribution: Optional[List[float]] = None
    state_feature_config_provider: ModelFeatureConfigProvider__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `raw`.
        # pyre-fixme[28]: Unexpected keyword argument `raw`.
        default_factory=lambda: ModelFeatureConfigProvider__Union(
            raw=RawModelFeatureConfigProvider(float_feature_infos=[])
        )
    )
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)
    preprocessing_options: Optional[PreprocessingOptions] = None
    reader_options: Optional[ReaderOptions] = None

    def __post_init_post_parse__(self):
        super().__init__()

    def create_policy(self, trainer) -> Policy:
        """ Create an online DiscreteDQN Policy from env. """
        sampler = SoftmaxActionSampler(temperature=self.trainer_param.rl.temperature)
        scorer = discrete_dqn_scorer(trainer.q_network)
        return Policy(scorer=scorer, sampler=sampler)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return self.state_feature_config_provider.value.get_model_feature_config()

    def metrics_to_score(self, reward_options: RewardOptions) -> List[str]:
        return get_metrics_to_score(reward_options.metric_reward_values)

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE]

    def run_feature_identification(
        self, data_fetcher: DataFetcher, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        preprocessing_options = self.preprocessing_options or PreprocessingOptions()
        logger.info("Overriding whitelist_features")
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        preprocessing_options = preprocessing_options._replace(
            whitelist_features=state_features
        )
        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=data_fetcher.identify_normalization_parameters(
                    input_table_spec, InputColumn.STATE_FEATURES, preprocessing_options
                )
            )
        }

    def query_data(
        self,
        data_fetcher: DataFetcher,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        return data_fetcher.query_data(
            input_table_spec=input_table_spec,
            discrete_action=True,
            actions=self.trainer_param.actions,
            include_possible_actions=True,
            sample_range=sample_range,
            custom_reward_expression=reward_options.custom_reward_expression,
            multi_steps=self.multi_steps,
            gamma=self.trainer_param.rl.gamma,
        )

    @property
    def multi_steps(self) -> Optional[int]:
        return self.trainer_param.rl.multi_steps

    def build_batch_preprocessor(
        self,
        reader_options: ReaderOptions,
        use_gpu: bool,
        batch_size: int,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            normalization_data_map[
                NormalizationKey.STATE
            ].dense_normalization_parameters,
            use_gpu=use_gpu,
        )
        return DiscreteDqnBatchPreprocessor(
            num_actions=len(self.trainer_param.actions),
            state_preprocessor=state_preprocessor,
            use_gpu=use_gpu,
        )

    def get_reporter(self):
        return DiscreteDQNReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )

    def get_evaluator(self, trainer, reward_options: RewardOptions):
        return Evaluator(
            self.trainer_param.actions,
            self.trainer_param.rl.gamma,
            trainer,
            metrics_to_score=self.metrics_to_score(reward_options),
        )
