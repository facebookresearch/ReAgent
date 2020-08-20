#!/usr/bin/env python3

import itertools
import logging
from typing import List, Tuple

from reagent.core import aggregators as agg
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.union import TrainingReport__Union
from reagent.reporting.oss_training_reports import (
    DebugToolsReport,
    OssWorldModelTrainingReport,
)
from reagent.reporting.reporter_base import ReporterBase


logger = logging.getLogger(__name__)


class WorldModelReporter(ReporterBase):
    def __init__(self, report_interval: int = 10):
        """
        For world model:
            'loss' (referring to total loss),
            'bce' (loss for predicting not_terminal),
            'gmm' (loss for next state prediction),
            'mse' (loss for predicting reward)
        """
        aggregators: List[Tuple[str, agg.Aggregator]] = list(
            itertools.chain(
                [
                    ("loss", agg.MeanAggregator("loss", interval=report_interval)),
                    ("bce", agg.MeanAggregator("bce", interval=report_interval)),
                    ("gmm", agg.MeanAggregator("gmm", interval=report_interval)),
                    ("mse", agg.MeanAggregator("mse", interval=report_interval)),
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(
                            key, log_key, interval=report_interval
                        ),
                    )
                    for key, log_key in [
                        ("loss", "loss"),
                        ("bce", "bce"),
                        ("gmm", "gmm"),
                        ("mse", "mse"),
                    ]
                ],
            )
        )
        super().__init__(aggregators)

    def publish(self) -> RLTrainingOutput:
        report = OssWorldModelTrainingReport(
            loss=self.loss.values,
            bce=self.bce.values,
            gmm=self.gmm.values,
            mse=self.mse.values,
        )
        return RLTrainingOutput(
            training_report=TrainingReport__Union(oss_world_model_report=report)
        )


class DebugToolsReporter(ReporterBase):
    def __init__(self, report_interval: int = 1):
        """
        For debug tools: feature_importance, feature_sensitivity
        """
        aggregators: List[Tuple[str, agg.Aggregator]] = [
            ("feature_importance", agg.AppendAggregator("feature_importance")),
            ("feature_sensitivity", agg.AppendAggregator("feature_sensitivity")),
        ]
        super().__init__(aggregators)

    def publish(self) -> RLTrainingOutput:
        feature_importance = (
            []
            if len(self.feature_importance.values) == 0
            else self.feature_importance.values[-1]
        )
        feature_sensitivity = (
            []
            if len(self.feature_sensitivity.values) == 0
            else self.feature_sensitivity.values[-1]
        )
        report = DebugToolsReport(
            feature_importance=feature_importance,
            feature_sensitivity=feature_sensitivity,
        )
        return RLTrainingOutput(
            training_report=TrainingReport__Union(oss_debug_tools_report=report)
        )
