#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import Evaluator, get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import (
    GreedyActionSampler,
    SoftmaxActionSampler,
)
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_scorer,
    discrete_dqn_serving_scorer,
)
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    DiscreteDqnBatchPreprocessor,
    InputColumn,
)
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.reporters.discrete_dqn_reporter import DiscreteDQNReporter
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


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorUnwrapper as DiscreteDqnPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import DiscreteDqnPredictorUnwrapper


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQNBase(ModelManager):
    target_action_distribution: Optional[List[float]] = None
    state_feature_config: rlt.ModelFeatureConfig = field(
        default_factory=lambda: rlt.ModelFeatureConfig(float_feature_infos=[])
    )
    preprocessing_options: Optional[PreprocessingOptions] = None
    reader_options: Optional[ReaderOptions] = None

    def __post_init_post_parse__(self):
        super().__init__()
        self._metrics_to_score = None
        self._q_network: Optional[ModelBase] = None

    def create_policy(self, serving: bool) -> Policy:
        """ Create an online DiscreteDQN Policy from env. """
        if serving:
            sampler = GreedyActionSampler()
            scorer = discrete_dqn_serving_scorer(
                DiscreteDqnPredictorUnwrapper(self.build_serving_module())
            )
        else:
            sampler = SoftmaxActionSampler(temperature=self.rl_parameters.temperature)
            # pyre-fixme[16]: `RLTrainer` has no attribute `q_network`.
            scorer = discrete_dqn_scorer(self.trainer.q_network)
        return Policy(scorer=scorer, sampler=sampler)

    @property
    def metrics_to_score(self) -> List[str]:
        assert self._reward_options is not None
        if self._metrics_to_score is None:
            # pyre-fixme[16]: `DiscreteDQNBase` has no attribute `_metrics_to_score`.
            # pyre-fixme[16]: `DiscreteDQNBase` has no attribute `_metrics_to_score`.
            self._metrics_to_score = get_metrics_to_score(
                # pyre-fixme[16]: `Optional` has no attribute `metric_reward_values`.
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

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
    ) -> RLTrainingOutput:
        """
        Train the model

        Returns partially filled RLTrainingOutput.
        The field that should not be filled are:
        - output_path
        """
        reporter = DiscreteDQNReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )
        # pyre-fixme[16]: `RLTrainer` has no attribute `add_observer`.
        self.trainer.add_observer(reporter)

        evaluator = Evaluator(
            self.action_names,
            self.rl_parameters.gamma,
            self.trainer,
            metrics_to_score=self.metrics_to_score,
        )
        # pyre-fixme[16]: `Evaluator` has no attribute `add_observer`.
        evaluator.add_observer(reporter)

        batch_preprocessor = self.build_batch_preprocessor()
        train_and_evaluate_generic(
            train_dataset,
            eval_dataset,
            # pyre-fixme[6]: Expected `RLTrainer` for 3rd param but got `Trainer`.
            # pyre-fixme[6]: Expected `RLTrainer` for 3rd param but got `Trainer`.
            self.trainer,
            num_epochs,
            self.use_gpu,
            batch_preprocessor,
            reporter,
            evaluator,
            reader_options=self.reader_options,
        )
        # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
        training_report = RLTrainingReport.make_union_instance(
            reporter.generate_training_report()
        )
        return RLTrainingOutput(training_report=training_report)
