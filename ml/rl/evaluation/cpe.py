#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
from ml.rl.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CpeEstimate(NamedTuple):
    raw: float
    normalized: float
    raw_std_error: float
    normalized_std_error: float


class CpeEstimateSet(NamedTuple):
    direct_method: Optional[CpeEstimate] = None
    inverse_propensity: Optional[CpeEstimate] = None
    doubly_robust: Optional[CpeEstimate] = None

    sequential_doubly_robust: Optional[CpeEstimate] = None
    weighted_doubly_robust: Optional[CpeEstimate] = None
    magic: Optional[CpeEstimate] = None

    def log(self):
        logger.info(
            "Reward Inverse Propensity Score : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.inverse_propensity.normalized,
                self.inverse_propensity.normalized_std_error,
                self.inverse_propensity.raw,
                self.inverse_propensity.raw_std_error,
            )
        )

        logger.info(
            "Reward Direct Method : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.direct_method.normalized,
                self.direct_method.normalized_std_error,
                self.direct_method.raw,
                self.direct_method.raw_std_error,
            )
        )

        logger.info(
            "Reward Doubly Robust P.E. : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.doubly_robust.normalized,
                self.doubly_robust.normalized_std_error,
                self.doubly_robust.raw,
                self.doubly_robust.raw_std_error,
            )
        )

        logger.info(
            "Value Weighted Doubly Robust P.E. : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.weighted_doubly_robust.normalized,
                self.weighted_doubly_robust.normalized_std_error,
                self.weighted_doubly_robust.raw,
                self.weighted_doubly_robust.raw_std_error,
            )
        )
        logger.info(
            "Value Sequential Doubly Robust P.E. : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.sequential_doubly_robust.normalized,
                self.sequential_doubly_robust.normalized_std_error,
                self.sequential_doubly_robust.raw,
                self.sequential_doubly_robust.raw_std_error,
            )
        )
        logger.info(
            "Value Magic Doubly Robust P.E. : normalized {0:.3f} +/- {0:.3f} raw {1:.3f} +/- {1:.3f}".format(
                self.magic.normalized,
                self.magic.normalized_std_error,
                self.magic.raw,
                self.magic.raw_std_error,
            )
        )

    def log_to_tensorboard(self, metric_name: str) -> None:
        def none_to_zero(x: Optional[float]) -> float:
            if x is None or math.isnan(x):
                return 0.0
            return x

        for name, value in [
            (
                "CPE/{}/Direct_Method_Reward".format(metric_name),
                self.direct_method.normalized,
            ),
            (
                "CPE/{}/IPS_Reward".format(metric_name),
                self.inverse_propensity.normalized,
            ),
            (
                "CPE/{}/Doubly_Robust_Reward".format(metric_name),
                self.doubly_robust.normalized,
            ),
            (
                "CPE/{}/Sequential_Doubly_Robust".format(metric_name),
                self.sequential_doubly_robust.normalized,
            ),
            (
                "CPE/{}/Weighted_Sequential_Doubly_Robust".format(metric_name),
                self.weighted_doubly_robust.normalized,
            ),
            ("CPE/{}/MAGIC".format(metric_name), self.magic.normalized),
        ]:
            SummaryWriterContext.add_scalar(name, none_to_zero(value))

    def fill_empty_with_zero(self):
        retval = self
        for name, value in self._asdict().items():
            if value is None:
                retval = retval._replace(
                    **{
                        name: CpeEstimate(
                            raw=0.0,
                            normalized=0.0,
                            raw_std_error=0.0,
                            normalized_std_error=0.0,
                        )
                    }
                )
        return retval


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


def bootstrapped_std_error_of_mean(data, sample_percent=0.25, num_samples=1000):
    """
    Compute bootstrapped standard error of mean of input data.

    :param data: Input data (1D torch tensor or numpy array).
    :param sample_percent: Size of sample to use to calculate bootstrap statistic.
    :param num_samples: Number of times to sample.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    sample_size = int(sample_percent * len(data))
    means = [
        np.mean(np.random.choice(data, size=sample_size, replace=True))
        for i in range(num_samples)
    ]
    return np.std(means)
