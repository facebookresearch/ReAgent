#!/usr/bin/env python3

import logging

from reagent.core import aggregators as agg
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.union import TrainingReport__Union
from reagent.reporting.oss_training_reports import OssRankingModelTrainingReport
from reagent.reporting.reporter_base import ReporterBase


logger = logging.getLogger(__name__)


class RankingModelReporter(ReporterBase):
    def __init__(self, report_interval: int = 100):
        """
        For Ranking model:
            'pg' (policy gradient loss)
            'baseline' (the baseline model's loss, usually for fitting V(s))
            'kendall_tau' (kendall_tau coefficient between advantage and log_probs,
             used in evaluation page handlers)
            'kendaull_tau_p_value' (the p-value for kendall_tau test, used in
             evaluation page handlers)
        """
        aggregators = [
            ("pg", agg.MeanAggregator("pg", interval=report_interval)),
            ("baseline", agg.MeanAggregator("baseline", interval=report_interval)),
            (
                "kendall_tau",
                agg.MeanAggregator("kendall_tau", interval=report_interval),
            ),
            (
                "kendaull_tau_p_value",
                agg.MeanAggregator("kendaull_tau_p_value", interval=report_interval),
            ),
        ] + [
            (
                f"{key}_tb",
                agg.TensorBoardHistogramAndMeanAggregator(
                    key, log_key, interval=report_interval
                ),
            )
            for key, log_key in [
                ("pg", "pg"),
                ("baseline", "baseline"),
                ("kendall_tau", "kendall_tau"),
                ("kendaull_tau_p_value", "kendaull_tau_p_value"),
            ]
        ]
        super().__init__(aggregators)

    # TODO: T71636236 write this for OSS
    def publish(self) -> RLTrainingOutput:
        report = OssRankingModelTrainingReport()
        return RLTrainingOutput(
            training_report=TrainingReport__Union(
                oss_ranking_model_training_report=report
            )
        )
