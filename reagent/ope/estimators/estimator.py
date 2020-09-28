#!/usr/bin/env python3

import logging
import math
import pickle
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import torch
from reagent.evaluation.cpe import bootstrapped_std_error_of_mean
from torch import Tensor


logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 1e-6


class ResultDiffs:
    """
    Statistics for differences, e.g., estimates vs ground truth
    """

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
            # pyre-fixme[16]: `Tensor` has no attribute `var`.
            self._variance = self._diffs.var()
        return self._variance

    def __repr__(self):
        return (
            f"samples={self._diffs.shape[0]}, rmse={self.rmse.item()}"
            f", bias={self.bias}, variance={self.variance}"
        )


@dataclass(frozen=True)
class EstimatorResult:
    log_reward: float
    estimated_reward: float
    ground_truth_reward: Optional[float] = 0.0
    estimated_weight: float = 1.0
    estimated_reward_normalized: Optional[float] = None
    estimated_reward_std_error: Optional[float] = None
    estimated_reward_normalized_std_error: Optional[float] = None


@dataclass
class EstimatorResults:
    """
    Estimator results
    """

    results: List[EstimatorResult] = field(default_factory=list)
    device = None

    def append(self, result: EstimatorResult):
        """Append a data point

        Args:
            result: result from an experimental run
        """
        er = result.estimated_reward
        if math.isnan(er) or math.isinf(er):
            logging.warning(f"  Invalid estimate: {er}")
            return
        lr = result.log_reward
        gr = (
            result.ground_truth_reward
            if result.ground_truth_reward is not None
            else 0.0
        )
        logging.info(
            f"  Append estimate [{len(self.results) + 1}]: "
            f"log={lr}, estimated={er}, ground_truth={gr}"
        )
        self.results.append(
            EstimatorResult(
                log_reward=result.log_reward,
                estimated_reward=result.estimated_reward,
                ground_truth_reward=gr,
                estimated_weight=result.estimated_weight,
            )
        )

    def report(self):
        ert = torch.tensor(
            [res.estimated_reward for res in self.results],
            dtype=torch.double,
            device=self.device,
        )
        lrt = torch.tensor(
            [res.log_reward for res in self.results],
            dtype=torch.double,
            device=self.device,
        )
        grt = torch.tensor(
            [
                res.ground_truth_reward if res.ground_truth_reward is not None else 0.0
                for res in self.results
            ],
            dtype=torch.double,
            device=self.device,
        )
        self._estimated_log_diff = ResultDiffs(ert - lrt)
        self._estimated_ground_truth_diff = ResultDiffs(ert - grt)
        return (
            lrt.mean().item(),
            ert.mean().item(),
            grt.mean().item(),
            ResultDiffs(ert - grt),
            ResultDiffs(ert - lrt),
            torch.tensor([float(res.estimated_weight) for res in self.results])
            .mean()
            .item(),
        )


@dataclass(frozen=True)
class EstimatorSampleResult:
    log_reward: float
    target_reward: float
    ground_truth_reward: float
    weight: float

    def __repr__(self):
        return (
            f"EstimatorSampleResult(log={self.log_reward}"
            f",tgt={self.target_reward},gt={self.ground_truth_reward}"
            f",wgt={self.weight}"
        )


class Estimator(ABC):
    """
    Estimator interface
    """

    def __init__(self, device=None):
        self._device = device

    def _compute_metric_data(
        self, tgt_rewards: Tensor, logged_score: float
    ) -> Tuple[float, float, float]:
        """
        Given a sequence of scores, normalizes the target score by the average logged score
        and computes the standard error of the target score. Normalizing by the logged score
        can provide a better metric to compare models against.
        """
        if len(tgt_rewards.shape) > 1:
            assert tgt_rewards.shape[1] == 1
            tgt_rewards = tgt_rewards.reshape((tgt_rewards.shape[0],))
        if logged_score < SCORE_THRESHOLD:
            normalizer = 0.0
        else:
            normalizer = 1.0 / logged_score
        std_err = bootstrapped_std_error_of_mean(tgt_rewards)
        return (
            torch.mean(tgt_rewards).item() * normalizer,
            std_err,
            std_err * normalizer,
        )

    @abstractmethod
    def evaluate(
        self, input, **kwargs
    ) -> Optional[Union[EstimatorResult, EstimatorResults]]:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(device({self._device}))"


def run_evaluation(
    file_name: str,
) -> Optional[Mapping[str, Iterable[EstimatorResults]]]:
    logger.info(f"received filename {file_name}")
    try:
        with open(file_name, "rb") as fp:
            estimators, inputs = pickle.load(fp)
    except Exception as err:
        return None
    results = {}
    for estimator in estimators:
        estimator_name = repr(estimator)
        estimator_results = []
        for input in inputs:
            try:
                estimator_results.append(estimator.evaluate(input))
            except Exception as err:
                logger.error(f"{estimator_name} error {err}")
        results[repr(estimator)] = estimator_results
    return results


class Evaluator:
    """
    Multiprocessing evaluator
    """

    def __init__(
        self,
        experiments: Iterable[Tuple[Iterable[Estimator], object]],
        max_num_workers: int,
    ):
        """
        Args:
            estimators: estimators to be evaluated
            experiments:
            max_num_workers: <= 0 no multiprocessing
                             otherwise create max_num_workers processes
        """
        self._experiments = experiments
        self._tasks = None
        if max_num_workers > 0:
            self._tasks = [[] for _ in range(max_num_workers)]
            for i, experiment in enumerate(experiments):
                self._tasks[i % max_num_workers].append(experiment)

    def evaluate(self) -> Mapping[str, EstimatorResults]:
        results = {}
        if self._tasks is None:
            for estimators, input in self._experiments:
                for estimator in estimators:
                    estimator_name = repr(estimator)
                    if estimator_name in results:
                        result = results[estimator_name]
                    else:
                        result = EstimatorResults()
                        results[estimator_name] = result
                    result.append(estimator.evaluate(input))
        else:
            tmp_files = []
            tmp_file_names = []
            for task in self._tasks:
                fp = tempfile.NamedTemporaryFile()
                pickle.dump(task, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.flush()
                tmp_files.append(fp)
                tmp_file_names.append(fp.name)
            with Pool(len(tmp_file_names)) as pool:
                evaluation_results = pool.map(run_evaluation, tmp_file_names)
            for tmp_file in tmp_files:
                tmp_file.close()
            for evaluation_result in evaluation_results:
                if evaluation_result is None:
                    continue
                for estimator_name, estimator_results in evaluation_result.items():
                    if estimator_name in results:
                        result = results[estimator_name]
                    else:
                        result = EstimatorResults()
                        results[estimator_name] = result
                    for estimator_result in estimator_results:
                        result.append(estimator_result)
        return results

    @staticmethod
    def report_results(results: Mapping[str, EstimatorResults]):
        for name, result in results.items():
            log_r, tgt_r, gt_r, tgt_gt, tgt_log, weight = result.report()
            print(
                f"{name} rewards: log_reward{log_r} tgt_reward[{tgt_r}] gt_reward[{gt_r}]"
                f", diffs: tgt-gt[{tgt_gt}] tgt-log[{tgt_log}]",
                flush=True,
            )
