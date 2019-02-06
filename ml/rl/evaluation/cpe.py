#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import Dict, NamedTuple, Optional

from ml.rl.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CpeEstimate(NamedTuple):
    raw: float
    normalized: float


class CpeEstimateSet:
    def __init__(self):
        self.direct_method: CpeEstimate = None
        self.inverse_propensity: CpeEstimate = None
        self.doubly_robust: CpeEstimate = None

        self.sequential_doubly_robust: CpeEstimate = None
        self.weighted_doubly_robust: CpeEstimate = None
        self.magic: CpeEstimate = None

    def log(self):
        logger.info(
            "Reward Inverse Propensity Score : normalized {0:.3f} raw {1:.3f}".format(
                self.inverse_propensity.normalized, self.inverse_propensity.raw
            )
        )

        logger.info(
            "Reward Direct Method : normalized {0:.3f} raw {1:.3f}".format(
                self.direct_method.normalized, self.direct_method.raw
            )
        )

        logger.info(
            "Reward Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                self.doubly_robust.normalized, self.doubly_robust.raw
            )
        )

        logger.info(
            "Value Weighted Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                self.weighted_doubly_robust.normalized, self.weighted_doubly_robust.raw
            )
        )
        logger.info(
            "Value Sequential Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                self.sequential_doubly_robust.normalized,
                self.sequential_doubly_robust.raw,
            )
        )
        logger.info(
            "Value Magic Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                self.magic.normalized, self.magic.raw
            )
        )

    def log_to_tensorboard(self, metric_name: str) -> None:
        def none_to_zero(x: Optional[float]) -> float:
            if x is None or math.isnan(x):
                return 0.0
            return x

        for name, value in [
            (
                "CPE/{}/Direct Method Reward".format(metric_name),
                self.direct_method.normalized,
            ),
            (
                "CPE/{}/IPS Reward".format(metric_name),
                self.inverse_propensity.normalized,
            ),
            (
                "CPE/{}/Doubly Robust Reward".format(metric_name),
                self.doubly_robust.normalized,
            ),
            (
                "CPE/{}/Sequential Doubly Robust".format(metric_name),
                self.sequential_doubly_robust.normalized,
            ),
            (
                "CPE/{}/Weighted Sequential Doubly Robust".format(metric_name),
                self.weighted_doubly_robust.normalized,
            ),
            ("CPE/{}/MAGIC".format(metric_name), self.magic.normalized),
        ]:
            SummaryWriterContext.add_scalar(name, none_to_zero(value))


class CpeDetails:
    def __init__(self):
        self.reward_estimates: CpeEstimateSet = CpeEstimateSet()
        self.metric_estimates: Dict[str, CpeEstimateSet] = {}
        self.mc_loss: float = None
        self.q_value_means: Optional[Dict[str, float]] = None
        self.q_value_stds: Optional[Dict[str, float]] = None
        self.action_distribution: Optional[Dict[str, float]] = None

    def log(self):
        logger.info("Reward Estimates:")
        logger.info("-----------------")
        self.reward_estimates.log()
        logger.info("-----------------")
        for metric in self.metric_estimates.keys():
            logger.info(metric + " Estimates:")
            logger.info("-----------------")
            self.metric_estimates[metric].log()
            logger.info("-----------------")

    def log_to_tensorboard(self) -> None:
        self.reward_estimates.log_to_tensorboard("Reward")
        for metric_name, estimate_set in self.metric_estimates.items():
            estimate_set.log_to_tensorboard(metric_name)
