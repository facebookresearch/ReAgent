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
    tgt_rewards: Sequence[Reward]


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
    TRAINING_VALIDATION_SPLIT = 0.8
    """
    Estimating using Direct Method (DM), assuming a reward model is trained
    """

    def __init__(self, trainer: Optional[Trainer] = None, device=None):
        super().__init__(device)
        self._trainer = trainer

    def _train_model(self, samples: Sequence[LogSample]) -> bool:
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
        training_size = int(sample_size * DMEstimator.TRAINING_VALIDATION_SPLIT)
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
    ) -> Tuple[Optional[Reward], torch.Tensor, torch.Tensor]:
        if sample.model_outputs is not None:
            return (
                sample.model_outputs.tgt_reward_from_log_action,
                torch.tensor(
                    sample.model_outputs.tgt_rewards,
                    dtype=torch.double,
                    device=self._device,
                ),
                torch.tensor(
                    # pyre-fixme[16]: `ActionDistribution` has no attribute `_values`.
                    sample.tgt_action_probabilities._values,
                    dtype=torch.double,
                    device=self._device,
                ),
            )
        trainer = self._trainer
        if trainer is None or not trainer.is_trained:
            return 0.0, torch.zeros(), torch.zeros()
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
            preds.scores,
            torch.tensor(probs, dtype=torch.double, device=self._device),
        )

    def _evaluate(
        self,
        input: BanditsEstimatorInput,
        train_samples: Sequence[LogSample],
        eval_samples: Sequence[LogSample],
        **kwargs,
    ) -> Optional[EstimatorResult]:
        if not self._train_model(train_samples) and not input.has_model_outputs:
            return None
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        tgt_vals = []
        gt_avg = RunningAverage()
        for sample in eval_samples:
            log_avg.add(sample.log_reward)
            _, tgt_scores, tgt_probs = self._calc_dm_reward(input.action_space, sample)
            tgt_reward = torch.dot(tgt_scores, tgt_probs).item()
            tgt_avg.add(tgt_reward)
            tgt_vals.append(tgt_reward)
            gt_avg.add(sample.ground_truth_reward)
        (
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(torch.tensor(tgt_vals), log_avg.average)
        return EstimatorResult(
            log_reward=log_avg.average,
            estimated_reward=tgt_avg.average,
            ground_truth_reward=gt_avg.average,
            estimated_weight=tgt_avg.count,
            estimated_reward_normalized=tgt_score_normalized,
            estimated_reward_std_error=tgt_std_err,
            estimated_reward_normalized_std_error=tgt_std_err_normalized,
        )

    @staticmethod
    def _calc_optional_avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
        # Annoying but Pyre would only take it like this
        return None if a is None else (None if b is None else (a + b) / 2)

    def evaluate(
        self, input: BanditsEstimatorInput, **kwargs
    ) -> Optional[EstimatorResult]:
        if input.has_model_outputs:
            return self._evaluate(input, input.samples, input.samples)
        log_avg = RunningAverage()
        gt_avg = RunningAverage()
        for sample in input.samples:
            log_avg.add(sample.log_reward)
            gt_avg.add(sample.ground_truth_reward)

        # 2-fold cross "validation" as used by https://arxiv.org/pdf/1612.01205.pdf
        shuffled = list(input.samples)
        np.random.shuffle(shuffled)
        lower_half = shuffled[: len(shuffled) // 2]
        upper_half = shuffled[len(shuffled) // 2 :]
        er_lower = self._evaluate(input, lower_half, upper_half)
        er_upper = self._evaluate(input, upper_half, lower_half)
        if er_lower is None or er_upper is None:
            return None
        return EstimatorResult(
            log_reward=log_avg.average,
            estimated_reward=(
                (er_lower.estimated_reward + er_upper.estimated_reward) / 2
            ),
            estimated_reward_normalized=(
                DMEstimator._calc_optional_avg(
                    er_lower.estimated_reward_normalized,
                    er_upper.estimated_reward_normalized,
                )
            ),
            estimated_reward_normalized_std_error=(
                DMEstimator._calc_optional_avg(
                    er_lower.estimated_reward_normalized_std_error,
                    er_upper.estimated_reward_normalized_std_error,
                )
            ),
            estimated_reward_std_error=(
                DMEstimator._calc_optional_avg(
                    er_lower.estimated_reward_std_error,
                    er_upper.estimated_reward_std_error,
                )
            ),
            ground_truth_reward=gt_avg.average,
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
            if sample.log_action.value is not None:
                weight = (
                    0.0
                    if sample.log_action_probabilities[sample.log_action]
                    < PROPENSITY_THRESHOLD
                    else sample.tgt_action_probabilities[sample.log_action]
                    / sample.log_action_probabilities[sample.log_action]
                )
                weight = self._weight_clamper(weight)
                tgt_result = sample.log_reward * weight
            tgt_avg.add(tgt_result)
            tgt_vals.append(tgt_result)
            acc_weight.add(weight)
            gt_avg.add(sample.ground_truth_reward)
        (
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(torch.tensor(tgt_vals), log_avg.average)
        return EstimatorResult(
            log_reward=log_avg.average,
            estimated_reward=tgt_avg.average
            if not self._weighted
            else tgt_avg.average / acc_weight.total,
            ground_truth_reward=gt_avg.average,
            estimated_weight=tgt_avg.count,
            estimated_reward_normalized=tgt_score_normalized,
            estimated_reward_std_error=tgt_std_err,
            estimated_reward_normalized_std_error=tgt_std_err_normalized,
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

    def _evaluate(
        self,
        input: BanditsEstimatorInput,
        train_samples: Sequence[LogSample],
        eval_samples: Sequence[LogSample],
        **kwargs,
    ) -> Optional[EstimatorResult]:
        self._train_model(train_samples)
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        tgt_vals = []
        gt_avg = RunningAverage()
        for sample in eval_samples:
            log_avg.add(sample.log_reward)
            dm_action_reward, dm_scores, dm_probs = self._calc_dm_reward(
                input.action_space, sample
            )
            dm_reward = torch.dot(dm_scores, dm_probs).item()
            tgt_result = 0.0
            weight = 0.0
            if sample.log_action.value is not None:
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
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(torch.tensor(tgt_vals), log_avg.average)
        return EstimatorResult(
            log_reward=log_avg.average,
            estimated_reward=tgt_avg.average,
            ground_truth_reward=gt_avg.average,
            estimated_weight=tgt_avg.count,
            estimated_reward_normalized=tgt_score_normalized,
            estimated_reward_std_error=tgt_std_err,
            estimated_reward_normalized_std_error=tgt_std_err_normalized,
        )

    def __repr__(self):
        return (
            f"DoublyRobustEstimator(trainer({self._trainer.name})"
            f",weight_clamper({self._weight_clamper}),device({self._device}))"
        )


class SwitchEstimator(DMEstimator):
    # For details, visit https://arxiv.org/abs/1612.01205 sections 4, 5
    CANDIDATES = 21
    EXP_BASE = 1.5

    def __init__(
        self,
        trainer: Optional[Trainer] = None,
        weight_clamper: Optional[Clamper] = None,
        rmax: Optional[Reward] = None,
        device=None,
    ):
        """
        rmax is an a priori upper bound on any possible reward.
        The tighter the bound, the better the estimator can estimate
        its bias. If not provided, the estimator will use the max
        reward seen in the sample data.
        """
        super().__init__(trainer, device)
        self._rmax = rmax
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper

    def _estimate_rmax(self, input: BanditsEstimatorInput) -> Reward:
        rmax = float("-inf")
        for sample in input.samples:
            _, dm_scores, dm_probs = self._calc_dm_reward(input.action_space, sample)
            max_sample_r = max(sample.log_reward, torch.max(dm_scores).item())
            rmax = max(rmax, max_sample_r)
        return rmax

    def _calc_weight_reward_tensors(
        self, input: BanditsEstimatorInput, eval_samples: Sequence[LogSample]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RunningAverage,
        RunningAverage,
    ]:
        n = len(eval_samples)
        ws = torch.ones((n, len(input.action_space)))
        rs = torch.zeros((n, 1))
        r_est = torch.zeros((n, len(input.action_space)))
        actions = torch.zeros((n, len(input.action_space)))
        expected_rmax = torch.zeros((n, len(input.action_space)))
        propensities = torch.zeros((n, len(input.action_space)))

        log_avg = RunningAverage()
        gt_avg = RunningAverage()

        priori_rmax = self._estimate_rmax(input) if self._rmax is None else self._rmax
        assert priori_rmax is not None

        for i, sample in enumerate(eval_samples):
            _, dm_scores, dm_probs = self._calc_dm_reward(input.action_space, sample)
            for a in input.action_space:
                weight = (
                    0.0
                    if sample.log_action_probabilities[a] < PROPENSITY_THRESHOLD
                    else sample.tgt_action_probabilities[a]
                    / sample.log_action_probabilities[a]
                )
                ws[i, a] = self._weight_clamper(weight)
                propensities[i, a] = sample.tgt_action_probabilities[a]
                expected_rmax[i, a] = sample.tgt_action_probabilities[a] * priori_rmax
                actions[i, a] = float(a == sample.log_action)

            rs[i, 0] = sample.log_reward
            r_est[i] = dm_scores
            log_avg.add(sample.log_reward)
            gt_avg.add(sample.ground_truth_reward)

        return actions, ws, rs, r_est, propensities, expected_rmax, log_avg, gt_avg

    def _calc_estimated_values(
        self,
        logged_rewards: torch.Tensor,
        weights: torch.Tensor,
        actions: torch.Tensor,
        threshold: float,
        est_rewards: torch.Tensor,
        tgt_props: torch.Tensor,
    ) -> torch.Tensor:
        ips_scores = (weights * actions).sum(dim=1, keepdim=True)
        return logged_rewards * ips_scores * (ips_scores <= threshold).float() + (
            est_rewards * tgt_props * (weights > threshold).float()
        ).sum(dim=1, keepdim=True)

    def _evaluate(
        self,
        input: BanditsEstimatorInput,
        train_samples: Sequence[LogSample],
        eval_samples: Sequence[LogSample],
        **kwargs,
    ) -> Optional[EstimatorResult]:
        self._train_model(train_samples)

        (
            actions,
            ws,
            rs,
            r_est,
            propensities,
            expected_rmax,
            log_avg,
            gt_avg,
        ) = self._calc_weight_reward_tensors(input, eval_samples)

        min_w, max_w = float(torch.min(ws).item()), float(torch.max(ws).item())
        diff = max_w - min_w

        # The threshold lies in the range [min ips, max ips]
        # Picking a small threshold -> using mainly the model-based estimator
        # Picking a large threshold -> using mainly the ips-based estimator
        candidates = [
            min_w
            + (
                (SwitchEstimator.EXP_BASE ** x)
                / (SwitchEstimator.EXP_BASE ** (SwitchEstimator.CANDIDATES - 1))
            )
            * diff
            for x in range(SwitchEstimator.CANDIDATES)
        ]
        tau = min_w
        loss = float("inf")
        for candidate in candidates:
            estimated_values = self._calc_estimated_values(
                rs, ws, actions, candidate, r_est, propensities
            )
            var = (1.0 / (estimated_values.shape[0] ** 2)) * torch.sum(
                (estimated_values - torch.mean(estimated_values)) ** 2
            ).item()
            bias = torch.mean(
                torch.sum(expected_rmax * (ws > candidate).float(), dim=1, keepdim=True)
            ).item()
            cand_loss = var + bias * bias
            if cand_loss < loss:
                tau = candidate
                loss = cand_loss

        estimated_values = self._calc_estimated_values(
            rs, ws, actions, tau, r_est, propensities
        )
        (
            tgt_score_normalized,
            tgt_std_err,
            tgt_std_err_normalized,
        ) = self._compute_metric_data(estimated_values, log_avg.average)
        return EstimatorResult(
            log_reward=log_avg.average,
            estimated_reward=torch.mean(estimated_values).item(),
            ground_truth_reward=gt_avg.average,
            estimated_weight=float(estimated_values.shape[0]),
            estimated_reward_normalized=tgt_score_normalized,
            estimated_reward_std_error=tgt_std_err,
            estimated_reward_normalized_std_error=tgt_std_err_normalized,
        )

    def __repr__(self):
        return (
            f"SwitchEstimator(trainer({self._trainer.name})"
            f",weight_clamper({self._weight_clamper}),device({self._device}))"
        )


class SwitchDREstimator(SwitchEstimator):
    # For details, visit https://arxiv.org/abs/1612.01205 sections 4, 5

    def _calc_estimated_values(
        self,
        logged_rewards: torch.Tensor,
        weights: torch.Tensor,
        actions: torch.Tensor,
        threshold: float,
        est_rewards: torch.Tensor,
        tgt_props: torch.Tensor,
    ) -> torch.Tensor:
        ips_scores = (weights * actions).sum(dim=1, keepdim=True)
        dr = ips_scores * (
            logged_rewards - (est_rewards * actions).sum(dim=1, keepdim=True)
        ) + (tgt_props * est_rewards).sum(dim=1, keepdim=True)
        return dr * (ips_scores <= threshold) + (
            est_rewards * tgt_props * (weights > threshold).float()
        ).sum(dim=1, keepdim=True)

    def __repr__(self):
        return (
            f"SwitchDREstimator(trainer({self._trainer.name})"
            f",weight_clamper({self._weight_clamper}),device({self._device}))"
        )
