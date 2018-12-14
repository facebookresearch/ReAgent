#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import Dict, NamedTuple, Optional

from tensorboardX import SummaryWriter


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

    def log_to_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        def none_to_zero(x: Optional[float]) -> float:
            if x is None or math.isnan(x):
                return 0.0
            return x

        for name, value in [
            ("Reward_CPE/Direct Method Reward", self.direct_method.normalized),
            ("Reward_CPE/IPS Reward", self.inverse_propensity.normalized),
            ("Reward_CPE/Doubly Robust Reward", self.doubly_robust.normalized),
            (
                "Value_CPE/Sequential Doubly Robust",
                self.sequential_doubly_robust.normalized,
            ),
            (
                "Value_CPE/Weighted Doubly Robust",
                self.weighted_doubly_robust.normalized,
            ),
            ("Value_CPE/MAGIC Estimator", self.magic.normalized),
        ]:
            writer.add_scalar(name, none_to_zero(value), epoch)


class CpeDetails:
    def __init__(self):
        self.reward_estimates: CpeEstimateSet = CpeEstimateSet()
        self.metric_estimates: Dict[str, CpeEstimateSet] = {}
        self.mc_loss: float = None

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
