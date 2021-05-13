#!/usr/bin/env python3
import copy
import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.models.base import ModelBase
from reagent.reporting.reporter_base import ReporterBase
from reagent.training.reward_network_trainer import LossFunction


logger = logging.getLogger(__name__)


class RewardNetworkReporter(ReporterBase):
    def __init__(
        self,
        loss_type: LossFunction,
        model_description: str,
        report_interval: int = 100,
    ):
        self.loss_type = loss_type
        self.model_description = model_description
        self.report_interval = report_interval
        self.best_model = None
        self.best_model_loss = float("inf")
        super().__init__(self.value_list_observers, self.aggregating_observers)

    @property
    def value_list_observers(self):
        return {}

    @property
    def aggregating_observers(self):
        return {
            name: IntervalAggregatingObserver(
                self.report_interval if "loss" in name else 1, aggregator
            )
            for name, aggregator in [
                ("loss", agg.MeanAggregator("loss")),
                ("unweighted_loss", agg.MeanAggregator("unweighted_loss")),
                ("eval_loss", agg.MeanAggregator("eval_loss")),
                ("eval_unweighted_loss", agg.MeanAggregator("eval_unweighted_loss")),
                ("eval_rewards", agg.EpochListAggregator("eval_rewards")),
                ("eval_pred_rewards", agg.EpochListAggregator("eval_pred_rewards")),
            ]
        }

    def update_best_model(self, loss: float, model: ModelBase):
        if loss < self.best_model_loss:
            self.best_model_loss = loss
            self.best_model = copy.deepcopy(model)
