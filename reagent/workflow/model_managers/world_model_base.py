#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import Evaluator, get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    DiscreteDqnBatchPreprocessor,
)
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


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorUnwrapper as DiscreteDqnPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import (  # type: ignore
        DiscreteDqnPredictorUnwrapper,
    )


logger = logging.getLogger(__name__)


@dataclass
class WorldModelBase(ModelManager):

    @classmethod
    def normalization_key(cls) -> str:
        return "state"

    def create_policy(self) -> Policy:
        """ Create a WorldModel Policy from env. """
        raise NotImplementedError()

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False

    def _set_normalization_parameters(
        self, normalization_data_map: Dict[str, NormalizationData]
    ):
        """
        Set normalization parameters on current instance
        """
        state_norm_data = normalization_data_map.get(self.normalization_key(), None)
        assert state_norm_data is not None
        assert state_norm_data.dense_normalization_parameters is not None
        self.state_normalization_parameters = (
            state_norm_data.dense_normalization_parameters
        )

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        raise NotImplementedError()

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError()

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        raise NotImplementedError()

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
    ) -> RLTrainingOutput:
        """
        Train the model

        Returns partially filled RLTrainingOutput. The field that should not be filled
        are:
        - output_path
        - warmstart_output_path
        - vis_metrics
        - validation_output
        """
        raise NotImplementedError()
