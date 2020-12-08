#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
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
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.reporters.discrete_dqn_reporter import DiscreteDQNReporter
from reagent.workflow.types import (
    Dataset,
    ModelFeatureConfigProvider__Union,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    RLTrainingReport,
    TableSpec,
)
from reagent.workflow.utils import train_eval_lightning


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQNBase(ModelManager):
    target_action_distribution: Optional[List[float]] = None
    state_feature_config_provider: ModelFeatureConfigProvider__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `raw`.
        default_factory=lambda: ModelFeatureConfigProvider__Union(
            raw=RawModelFeatureConfigProvider(float_feature_infos=[])
        )
    )
    preprocessing_options: Optional[PreprocessingOptions] = None
    reader_options: Optional[ReaderOptions] = None
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init_post_parse__(self):
        super().__init__()
        self._metrics_to_score = None
        self._q_network: Optional[ModelBase] = None

    def create_policy(self, serving: bool) -> Policy:
        """ Create an online DiscreteDQN Policy from env. """
        if serving:
            return create_predictor_policy_from_model(
                self.build_serving_module(), rl_parameters=self.rl_parameters
            )
        else:
            sampler = SoftmaxActionSampler(temperature=self.rl_parameters.temperature)
            # pyre-fixme[16]: `RLTrainer` has no attribute `q_network`.
            scorer = discrete_dqn_scorer(self.trainer.q_network)
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return self.state_feature_config_provider.value.get_model_feature_config()

    @property
    def metrics_to_score(self) -> List[str]:
        assert self._reward_options is not None
        if self._metrics_to_score is None:
            # pyre-fixme[16]: `DiscreteDQNBase` has no attribute `_metrics_to_score`.
            self._metrics_to_score = get_metrics_to_score(
                # pyre-fixme[16]: `Optional` has no attribute `metric_reward_values`.
                self._reward_options.metric_reward_values
            )
        return self._metrics_to_score

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE]

    def run_feature_identification(
        self, input_table_spec: TableSpec
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
                dense_normalization_parameters=identify_normalization_parameters(
                    input_table_spec, InputColumn.STATE_FEATURES, preprocessing_options
                )
            )
        }

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        return query_data(
            input_table_spec=input_table_spec,
            discrete_action=True,
            actions=self.action_names,
            include_possible_actions=True,
            sample_range=sample_range,
            custom_reward_expression=reward_options.custom_reward_expression,
            multi_steps=self.multi_steps,
            gamma=self.rl_parameters.gamma,
        )

    @property
    def multi_steps(self) -> Optional[int]:
        return self.rl_parameters.multi_steps

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            self.state_normalization_data.dense_normalization_parameters,
            use_gpu=self.use_gpu,
        )
        return DiscreteDqnBatchPreprocessor(
            num_actions=len(self.action_names),
            state_preprocessor=state_preprocessor,
            use_gpu=self.use_gpu,
        )

    def get_reporter(self):
        return DiscreteDQNReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )

    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
    ) -> RLTrainingOutput:
        """
        Train the model

        Returns partially filled RLTrainingOutput.
        The field that should not be filled are:
        - output_path
        """
        batch_preprocessor = self.build_batch_preprocessor()
        reporter = self.get_reporter()
        # pyre-fixme[16]: `RLTrainer` has no attribute `set_reporter`.
        self.trainer.set_reporter(reporter)

        train_eval_lightning(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            trainer_module=self.trainer,
            data_module=None,
            num_epochs=num_epochs,
            use_gpu=self.use_gpu,
            batch_preprocessor=batch_preprocessor,
            reader_options=self.reader_options,
            checkpoint_path=self._lightning_checkpoint_path,
        )
        # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
        training_report = RLTrainingReport.make_union_instance(
            reporter.generate_training_report()
        )
        return RLTrainingOutput(training_report=training_report)
