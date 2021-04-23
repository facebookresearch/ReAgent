#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
)
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import (
    GreedyActionSampler,
)
from reagent.gym.policies.scorers.discrete_scorer import discrete_dqn_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.models.model_feature_config_provider import RawModelFeatureConfigProvider
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    DiscreteDqnBatchPreprocessor,
)
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.preprocessing.types import InputColumn
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.reporters.discrete_dqn_reporter import DiscreteDQNReporter
from reagent.workflow.types import (
    Dataset,
    ModelFeatureConfigProvider__Union,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    RLTrainingReport,
    TableSpec,
)
from reagent.workflow.utils import train_eval_lightning, get_rank


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
        super().__post_init_post_parse__()
        self._metrics_to_score = None
        self._q_network: Optional[ModelBase] = None

    def create_policy(self, serving: bool) -> Policy:
        """ Create an online DiscreteDQN Policy from env. """
        if serving:
            return create_predictor_policy_from_model(
                self.build_serving_module(), rl_parameters=self.rl_parameters
            )
        else:
            sampler = GreedyActionSampler()
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
                self._reward_options.metric_reward_values
            )
        return self._metrics_to_score

    @property
    def should_generate_eval_dataset(self) -> bool:
        raise RuntimeError

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE]

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        raise RuntimeError

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise RuntimeError

    @property
    def multi_steps(self) -> Optional[int]:
        return self.rl_parameters.multi_steps

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        raise RuntimeError

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        reader_options: Optional[ReaderOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
    ) -> Optional[ReAgentDataModule]:
        return DiscreteDqnDataModule(
            input_table_spec=input_table_spec,
            reward_options=reward_options,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            reader_options=reader_options,
            model_manager=self,
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
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions] = None,
    ) -> RLTrainingOutput:
        """
        Train the model

        Returns partially filled RLTrainingOutput.
        The field that should not be filled are:
        - output_path
        """
        reporter = self.get_reporter()
        # pyre-fixme[16]: `RLTrainer` has no attribute `set_reporter`.
        self.trainer.set_reporter(reporter)
        assert data_module

        # pyre-fixme[16]: `DiscreteDQNBase` has no attribute `_lightning_trainer`.
        self._lightning_trainer = train_eval_lightning(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            trainer_module=self.trainer,
            data_module=data_module,
            num_epochs=num_epochs,
            use_gpu=self.use_gpu,
            logger_name="DiscreteDqn",
            reader_options=reader_options,
            checkpoint_path=self._lightning_checkpoint_path,
            resource_options=resource_options,
        )
        rank = get_rank()
        if rank == 0:
            # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
            training_report = RLTrainingReport.make_union_instance(
                reporter.generate_training_report()
            )
            logger_data = self._lightning_trainer.logger.line_plot_aggregated
            self._lightning_trainer.logger.clear_local_data()
            return RLTrainingOutput(
                training_report=training_report, logger_data=logger_data
            )
        # Output from processes with non-0 rank is not used
        return RLTrainingOutput()


class DiscreteDqnDataModule(ManualDataModule):
    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.calc_cpe_in_training

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE]

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        preprocessing_options = (
            self.model_manager.preprocessing_options or PreprocessingOptions()
        )
        logger.info("Overriding allowedlist_features")
        state_features = [
            ffi.feature_id
            for ffi in self.model_manager.state_feature_config.float_feature_infos
        ]
        preprocessing_options = preprocessing_options._replace(
            allowedlist_features=state_features
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
        data_fetcher: DataFetcher,
    ) -> Dataset:
        return data_fetcher.query_data(
            input_table_spec=input_table_spec,
            discrete_action=True,
            actions=self.model_manager.action_names,
            include_possible_actions=True,
            sample_range=sample_range,
            custom_reward_expression=reward_options.custom_reward_expression,
            multi_steps=self.model_manager.multi_steps,
            gamma=self.model_manager.rl_parameters.gamma,
        )

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            self.state_normalization_data.dense_normalization_parameters,
            use_gpu=self.model_manager.use_gpu,
        )
        return DiscreteDqnBatchPreprocessor(
            num_actions=len(self.model_manager.action_names),
            state_preprocessor=state_preprocessor,
            use_gpu=self.model_manager.use_gpu,
        )
