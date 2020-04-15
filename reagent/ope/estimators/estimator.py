#!/usr/bin/env python3

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class ResultDiffs:
    def __init__(self, diffs: Tensor):
        self._diffs = diffs
        self._rmse = None
        self._bias = None
        self._variance = None

    @property
    def rmse(self) -> Tensor:
        if self._rmse is None:
            self._rmse = (self._diffs ** 2.0).mean().sqrt()
        return self._rmse

    @property
    def bias(self) -> Tensor:
        if self._bias is None:
            self._bias = self._diffs.mean()
        return self._bias

    @property
    def variance(self) -> Tensor:
        if self._variance is None:
            self._variance = self._diffs.var()
        return self._variance

    def __repr__(self):
        return (
            f"samples={self._diffs.shape[0]}, rmse={self.rmse.item()}"
            f", bias={self.bias}, variance={self.variance}"
        )


@dataclass(frozen=True)
class EstimatorResults:
    """
    Estimator results
    """

    logs: Tensor
    estimates: Tensor
    ground_truths: Optional[Tensor] = None
    estimate_log_diffs: Optional[ResultDiffs] = None
    estimate_gt_diffs: Optional[ResultDiffs] = None

    def __repr__(self):
        repr = ""
        if self.estimate_gt_diffs is not None:
            repr += f"Target vs GT: {self.estimate_gt_diffs}"
        if self.estimate_log_diffs is not None:
            if len(repr) > 0:
                repr += ", "
            repr += f"Target vs Log: {self.estimate_log_diffs}"
        return repr


class Estimator(ABC):
    """
    Estimator interface
    """

    def __init__(self, device=None):
        self._device = device
        self._logs = []  # logged values
        self._estimates = []  # estimated values
        self._ground_truths = []  # ground truth values
        self._results = None

    def reset(self):
        self._logs.clear()
        self._estimates.clear()
        self._ground_truths.clear()
        self._results = None

    @property
    def logged_values(self):
        return self._logs

    @property
    def estimated_values(self):
        return self._estimates

    @property
    def ground_truth_values(self):
        return self._ground_truths

    def _append_estimate(
        self,
        log: Union[float, Tensor],
        estimate: Union[float, Tensor],
        ground_truth: Optional[Union[float, Tensor]] = None,
    ):
        if math.isnan(estimate) or math.isinf(estimate):
            return
        logging.info(
            f"  Append estimate [{len(self._estimates) + 1}]: "
            f"{log}, {estimate}, {ground_truth}"
        )
        self._logs.append(log)
        self._estimates.append(estimate)
        if ground_truth is not None:
            self._ground_truths.append(ground_truth)
        self._results = None

    @property
    def results(self) -> EstimatorResults:
        if self._results is None:
            logs_tensor = torch.tensor(
                self._logs, dtype=torch.float, device=self._device
            )
            estimates_tensor = torch.tensor(
                self._estimates, dtype=torch.float, device=self._device
            )
            if len(self._ground_truths) == len(self._estimates):
                ground_truths_tensor = torch.tensor(
                    self._ground_truths, dtype=torch.float, device=self._device
                )
                log_gt_diff = logs_tensor - ground_truths_tensor
            else:
                ground_truths_tensor = None
                log_gt_diff = None
            self._results = EstimatorResults(
                logs_tensor,
                estimates_tensor,
                ground_truths_tensor,
                ResultDiffs(log_gt_diff) if log_gt_diff is not None else None,
                ResultDiffs(estimates_tensor - logs_tensor),
            )
        return self._results

    @abstractmethod
    def evaluate(self, input, **kwargs) -> EstimatorResults:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}{{device[{self._device}]}}"
