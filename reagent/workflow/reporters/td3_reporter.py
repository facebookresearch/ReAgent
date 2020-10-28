#!/usr/bin/env python3

import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.workflow.reporters.actor_critic_reporter import ActorCriticReporter


logger = logging.getLogger(__name__)


class TD3Reporter(ActorCriticReporter):
    @property
    def aggregating_observers(self):
        ret = super().aggregating_observers
        ret.update(
            {
                name: IntervalAggregatingObserver(1, aggregator)
                for name, aggregator in [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                    )
                    for key, log_key in [
                        ("q1_loss", "loss/q1_loss"),
                        ("actor_loss", "loss/actor_loss"),
                        ("q1_value", "q_value/q1_value"),
                        ("next_q_value", "q_value/next_q_value"),
                        ("target_q_value", "q_value/target_q_value"),
                        ("actor_q1_value", "q_value/actor_q1_value"),
                        ("q2_loss", "loss/q2_loss"),
                        ("q2_value", "q_value/q2_value"),
                    ]
                ]
            }
        )
        return ret
