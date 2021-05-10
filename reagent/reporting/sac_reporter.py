#!/usr/bin/env python3

import logging

from reagent.core import aggregators as agg
from reagent.core.observers import (
    IntervalAggregatingObserver,
    TensorBoardScalarObserver,
)
from reagent.reporting.actor_critic_reporter import ActorCriticReporter


logger = logging.getLogger(__name__)


class SACReporter(ActorCriticReporter):
    @property
    def value_list_observers(self):
        ret = super().value_list_observers
        ret.update(
            {
                f"{key}_tb": TensorBoardScalarObserver(key, log_key)
                for key, log_key in [("entropy_temperature", None), ("kld", "kld/kld")]
            }
        )
        return ret

    @property
    def aggregating_observers(self):
        ret = super().aggregating_observers
        ret.update({})
        ret.update(
            {
                name: IntervalAggregatingObserver(1, aggregator)
                for name, aggregator in [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                    )
                    for key, log_key in [
                        ("q1_value", "q1/logged_state_value"),
                        ("q2_value", "q2/logged_state_value"),
                        ("log_prob_a", "log_prob_a"),
                        ("target_state_value", "value_network/target"),
                        ("next_state_value", "q_network/next_state_value"),
                        ("target_q_value", "q_network/target_q_value"),
                        ("actor_output_log_prob", "actor/log_prob"),
                        ("min_q_actor_value", "actor/min_q_actor_value"),
                        ("actor_loss", "actor/loss"),
                        ("action_batch_mean", "kld/mean"),
                        ("action_batch_var", "kld/var"),
                        ("entropy_temperature", "entropy_temperature"),
                    ]
                ]
            }
        )
        return ret
