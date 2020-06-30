#!/usr/bin/env python3

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

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


logger = logging.getLogger(__name__)
Actions = Union[Sequence[Action], Tensor, np.ndarray]
PROPENSITY_THRESHOLD = 1e-6


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
class ModelOutputs:
    tgt_reward_from_log_action: Reward
    tgt_rewards: Reward


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
    tgt_action: Action
    model_outputs: Optional[ModelOutputs] = None
    ground_truth_reward: Reward = float("nan")
    item_feature: Optional[Tensor] = None


@dataclass(frozen=True)
class BanditsEstimatorInput:
    action_space: ActionSpace
    samples: Sequence[LogSample]
    has_model_outputs: bool


class DMEstimator(Estimator):
    """
    Estimating using Direct Method (DM), assuming a reward model is trained
    """

    def __init__(self, trainer: Optional[Trainer] = None, device=None):
        super().__init__(device)
        self._trainer = trainer

    def _train_model(self, samples: Sequence[LogSample], ratio: float) -> bool:
        if self._trainer is None:
            logger.error("Target model trainer not set")
            return False
        trainer = self._trainer
        assert trainer is not None
        if trainer.is_trained:
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
        trainer.train(training_data)
        logger.info(f"  training direct model done: {time.perf_counter() - st}s")
        return True

    def _calc_dm_reward(
        self, action_space: ActionSpace, sample: LogSample
    ) -> Tuple[Optional[Reward], Optional[Reward]]:
        if sample.model_outputs is not None:
            return (
                sample.model_outputs.tgt_reward_from_log_action,
                torch.dot(
                    torch.tensor(
                        sample.model_outputs.tgt_rewards,
                        dtype=torch.double,
                        device=self._device,
                    ),
                    torch.tensor(
                        sample.tgt_action_probabilities,
                        dtype=torch.double,
                        device=self._device,
                    ),
                ),
            )
        trainer = self._trainer
        if trainer is None or not trainer.is_trained:
            return 0.0, 0.0
        assert sample.item_feature is not None
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
        preds = trainer.predict(torch.stack(features), device=self._device)
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
        if not self._train_model(input.samples, 0.8) and not input.has_model_outputs:
            return None
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        tgt_vals = []
        logged_vals = []
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            logged_vals.append(sample.log_reward)
            _, tgt_reward = self._calc_dm_reward(input.action_space, sample)
            tgt_avg.add(tgt_reward)
            tgt_vals.append(tgt_reward)
            gt_avg.add(sample.ground_truth_reward)
        (
            tgt_score,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(
            torch.tensor(tgt_vals), torch.tensor(logged_vals), tgt_avg.average
        )
        return EstimatorResult(
            log_avg.average,
            tgt_score,
            gt_avg.average,
            tgt_avg.count,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        )

    def __repr__(self):
        return f"DMEstimator(trainer({self._trainer.name},device({self._device}))"


class IPSEstimator(Estimator):
    """
    Inverse Propensity Scoring (IPS) estimator
    """

    def __init__(
        self,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = False,
        device=None,
    ):
        super().__init__(device)
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper
        self._weighted = weighted

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        log_avg = RunningAverage()
        logged_vals = []
        tgt_avg = RunningAverage()
        tgt_vals = []
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            logged_vals.append(sample.log_reward)
            weight = 0.0
            tgt_result = 0.0
            if sample.log_action is not None:
                weight = (
                    sample.tgt_action_probabilities[sample.log_action]
                    / sample.log_action_probabilities[sample.log_action]
                )
                weight = self._weight_clamper(weight)
                tgt_result = sample.log_reward * weight
            tgt_avg.add(tgt_result)
            tgt_vals.append(tgt_result)
            acc_weight.add(weight)
            gt_avg.add(sample.ground_truth_reward)
        (
            tgt_score,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(
            torch.tensor(tgt_vals), torch.tensor(logged_vals), tgt_avg.average
        )
        return EstimatorResult(
            log_avg.average,
            tgt_score if not self._weighted else tgt_score / acc_weight.total,
            gt_avg.average,
            tgt_avg.count,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
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
        self,
        trainer: Optional[Trainer] = None,
        weight_clamper: Optional[Clamper] = None,
        device=None,
    ):
        super().__init__(trainer, device)
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        self._train_model(input.samples, 0.8)
        log_avg = RunningAverage()
        logged_vals = []
        tgt_avg = RunningAverage()
        tgt_vals = []
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            logged_vals.append(sample.log_reward)
            dm_action_reward, dm_reward = self._calc_dm_reward(
                input.action_space, sample
            )
            tgt_result = 0.0
            weight = 0.0
            if sample.log_action is not None:
                weight = (
                    0.0
                    if sample.log_action_probabilities[sample.log_action]
                    < PROPENSITY_THRESHOLD
                    else sample.tgt_action_probabilities[sample.log_action]
                    / sample.log_action_probabilities[sample.log_action]
                )
                weight = self._weight_clamper(weight)
                assert dm_action_reward is not None
                assert dm_reward is not None
                tgt_result += (
                    sample.log_reward - dm_action_reward
                ) * weight + dm_reward
            else:
                tgt_result = dm_reward
            tgt_avg.add(tgt_result)
            tgt_vals.append(tgt_result)
            gt_avg.add(sample.ground_truth_reward)
        (
            tgt_score,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(
            torch.tensor(tgt_vals), torch.tensor(logged_vals), tgt_avg.average
        )
        return EstimatorResult(
            log_avg.average,
            tgt_score,
            gt_avg.average,
            tgt_avg.count,
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        )

    def __repr__(self):
        return (
            f"DoublyRobustEstimator(trainer({self._trainer.name})"
            f",weight_clamper({self._weight_clamper}),device({self._device}))"
        )
