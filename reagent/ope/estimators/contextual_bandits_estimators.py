#!/usr/bin/env python3

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from reagent.ope.estimators.estimator import Estimator, EstimatorResult
from reagent.ope.estimators.types import (
    Action,
    ActionDistribution,
    ActionSpace,
    Reward,
    Trainer,
    TrainingData,
    Values,
)
from reagent.ope.utils import Clamper, RunningAverage
from torch import Tensor


Actions = Union[Sequence[Action], Tensor, np.ndarray]


class ActionRewards(Values[Action]):
    def _to_key(self, k: int) -> Action:
        return Action(k)


class BanditsModel(ABC):
    @abstractmethod
    def _action_rewards(self, context) -> ActionRewards:
        """
        Calculate reward for each action based on context
        Args:
            context: task specific context

        Returns: action -> reward map

        """
        pass

    def __call__(self, context) -> ActionRewards:
        return self._action_rewards(context)


@dataclass(frozen=True)
class LogSample:
    # task specific context
    context: object
    # log
    log_action: Action
    log_reward: Reward
    log_action_probabilities: ActionDistribution
    # result from target policy
    tgt_action_probabilities: ActionDistribution
    ground_truth_reward: Reward = float("nan")
    item_feature: Tensor = None


@dataclass(frozen=True)
class BanditsEstimatorInput:
    action_space: ActionSpace
    samples: Sequence[LogSample]


class DMEstimator(Estimator):
    """
    Estimating using Direct Method (DM), assuming a reward model is trained
    """

    def __init__(self, trainer: Trainer, device=None):
        super().__init__(device)
        self._trainer = trainer

    def _train_model(
        self, samples: Sequence[LogSample], ratio: float, logger: logging.Logger
    ) -> bool:
        if self._trainer is None:
            logger.error("Target model trainer not set")
            return False
        if self._trainer.is_trained:
            return True
        logger.info("  training direct model...")
        st = time.perf_counter()
        sample_size = len(samples)
        if ratio > 0.0 and ratio < 1.0:
            training_size = int(sample_size * ratio)
        else:
            training_size = sample_size
        train_x = []
        train_y = []
        for i in range(training_size):
            sample = samples[i]
            if sample.item_feature is None:
                continue
            train_x.append(
                torch.cat(
                    (
                        torch.tensor(
                            sample.log_action.value, dtype=torch.float
                        ).flatten(),
                        sample.item_feature.flatten(),
                    )
                )
            )
            train_y.append(sample.log_reward)
        if len(train_x) == 0:
            logger.error("Item features not provided, DM is not available")
            return False
        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y, dtype=torch.double, device=train_x.device)
        vali_x = []
        vali_y = []
        for i in range(training_size, sample_size):
            sample = samples[i]
            if sample.item_feature is None:
                continue
            vali_x.append(
                torch.cat(
                    (
                        torch.tensor(
                            sample.log_action.value, dtype=torch.float
                        ).flatten(),
                        sample.item_feature.flatten(),
                    )
                )
            )
            vali_y.append(sample.log_reward)
        if len(vali_x) == 0:
            vali_x = train_x.detach().clone()
            vali_y = train_y.detach().clone()
        else:
            vali_x = torch.stack(vali_x)
            vali_y = torch.tensor(vali_y, dtype=torch.double, device=vali_x.device)
        training_data = TrainingData(train_x, train_y, None, vali_x, vali_y, None)
        self._trainer.train(training_data)
        logger.info(f"  training direct model done: {time.perf_counter() - st}s")
        return True

    def _calc_dm_reward(
        self, action_space: ActionSpace, sample: LogSample
    ) -> Tuple[Reward, Reward]:
        if self._trainer is None or not self._trainer.is_trained:
            return 0.0, 0.0
        item_feature = sample.item_feature.flatten()
        features = []
        probs = []
        idx = -1
        for action in action_space:
            if idx < 0 and action == sample.log_action:
                idx = len(features)
            features.append(
                torch.cat(
                    (
                        torch.tensor(action.value, dtype=torch.float).flatten(),
                        item_feature,
                    )
                )
            )
            probs.append(sample.tgt_action_probabilities[action])
        preds = self._trainer.predict(torch.stack(features), device=self._device)
        return (
            preds.scores[idx].item(),
            torch.dot(
                preds.scores,
                torch.tensor(probs, dtype=torch.double, device=self._device),
            ).item(),
        )

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        logger = Estimator.logger()
        if not self._train_model(input.samples, 0.8, logger):
            return None
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            _, tgt_reward = self._calc_dm_reward(input.action_space, sample)
            tgt_avg.add(tgt_reward)
            gt_avg.add(sample.ground_truth_reward)
        return EstimatorResult(
            log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
        )

    def __repr__(self):
        return f"DMEstimator(trainer({self._trainer.name},device({self._device}))"


class IPSEstimator(Estimator):
    """
    Inverse Propensity Scoring (IPS) estimator
    """

    def __init__(
        self, weight_clamper: Clamper = None, weighted: bool = False, device=None
    ):
        super().__init__(device)
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper
        self._weighted = weighted

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            weight = (
                sample.tgt_action_probabilities[sample.log_action]
                / sample.log_action_probabilities[sample.log_action]
            )
            weight = self._weight_clamper(weight)
            tgt_avg.add(sample.log_reward * weight)
            acc_weight.add(weight)
            gt_avg.add(sample.ground_truth_reward)
        if self._weighted:
            return EstimatorResult(
                log_avg.average,
                tgt_avg.total / acc_weight.total,
                gt_avg.average,
                acc_weight.average,
            )
        else:
            return EstimatorResult(
                log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
            )

    def __repr__(self):
        return (
            f"IPSEstimator(weight_clamper({self._weight_clamper})"
            f",weighted({self._weighted}),device({self._device}))"
        )


class DoublyRobustEstimator(DMEstimator):
    """
    Doubly Robust (DR) estimator:
        reference: https://arxiv.org/abs/1103.4601 (deterministic reward model)
                   https://arxiv.org/abs/1612.01205 (distributed reward model)
    """

    def __init__(
        self, trainer: Trainer = None, weight_clamper: Clamper = None, device=None
    ):
        super().__init__(trainer, device)
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        logger = Estimator.logger()
        self._train_model(input.samples, 0.8, logger)
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            weight = (
                sample.tgt_action_probabilities[sample.log_action]
                / sample.log_action_probabilities[sample.log_action]
            )
            weight = self._weight_clamper(weight)
            dm_action_reward, dm_reward = self._calc_dm_reward(
                input.action_space, sample
            )
            tgt_avg.add((sample.log_reward - dm_action_reward) * weight + dm_reward)
            gt_avg.add(sample.ground_truth_reward)
        return EstimatorResult(
            log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
        )

    def __repr__(self):
        return (
            f"DoublyRobustEstimator(trainer({self._trainer.name})"
            f",weight_clamper({self._weight_clamper}),device({self._device}))"
        )
