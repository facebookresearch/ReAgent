#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from reagent.ope.estimators.estimator import Estimator, EstimatorResults
from reagent.ope.estimators.types import (
    Action,
    ActionDistribution,
    ActionSpace,
    Reward,
    Values,
)
from reagent.ope.utils import Clamper, RunningAverage
from torch import Tensor


Actions = Union[Sequence[Action], Tensor, np.ndarray]


class ActionRewards(Values[Action]):
    def _new_key(self, k: int) -> Action:
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
    logged_action: Action
    logged_propensities: ActionDistribution
    logged_reward: Reward
    # result from target policy
    target_action: Action
    target_propensities: ActionDistribution


@dataclass(frozen=True)
class Log:
    """
    Input for contextual bandits estimators
        Tensor is used if action can be indexed in [0, action_space)
        Otherwise, Sequence and Mapping are used
    """

    samples: Iterable[LogSample]


@dataclass(frozen=True)
class BanditsEstimatorInput:
    action_space: ActionSpace
    logs: Iterable[Log]
    target_model: Optional[BanditsModel] = None
    ground_truth_model: Optional[BanditsModel] = None


class DMEstimator(Estimator):
    """
    Estimating using Direct Method (DM), assuming a reward model is trained
    """

    def evaluate(self, input: BanditsEstimatorInput, **kwargs) -> EstimatorResults:
        self.reset()
        for log in input.logs:
            log_reward = RunningAverage()
            tgt_reward = RunningAverage()
            gt_reward = RunningAverage()
            for sample in log.samples:
                log_reward.add(sample.logged_reward)
                rewards = input.target_model(sample.context)
                tgt_reward.add(rewards[sample.target_action])
                rewards = input.ground_truth_model(sample.context)
                gt_reward.add(rewards[sample.target_action])
            self._append_estimate(
                log_reward.average, tgt_reward.average, gt_reward.average
            )
        return self.results


class IPSEstimator(Estimator):
    """
    Inverse Propensity Scoring (IPS) estimator
    """

    def __init__(self, weight_clamper: Clamper = None, device=None):
        super().__init__(device)
        self._weight_clamper = Clamper() if weight_clamper is None else weight_clamper

    def evaluate(self, input: BanditsEstimatorInput, **kwargs) -> EstimatorResults:
        self.reset()
        for log in input.logs:
            log_reward = RunningAverage()
            tgt_reward = RunningAverage()
            gt_reward = RunningAverage()
            for sample in log.samples:
                log_reward.add(sample.logged_reward)
                weight = (
                    sample.target_propensities[sample.logged_action]
                    / sample.logged_propensities[sample.logged_action]
                )
                weight = self._weight_clamper(weight)
                tgt_reward.add(sample.logged_reward * weight)
                rewards = input.ground_truth_model(sample.context)
                gt_reward.add(rewards[sample.target_action])
            self._append_estimate(
                log_reward.average, tgt_reward.average, gt_reward.average
            )
        return self.results


class DoublyRobustEstimator(IPSEstimator):
    """
    Doubly Robust (DR) estimator:
        reference: https://arxiv.org/abs/1103.4601 (deterministic reward model)
                   https://arxiv.org/abs/1612.01205 (distributed reward model)
    """

    def evaluate(self, input: BanditsEstimatorInput, **kwargs) -> EstimatorResults:
        self.reset()
        for log in input.logs:
            log_reward = RunningAverage()
            tgt_reward = RunningAverage()
            gt_reward = RunningAverage()
            for sample in log.samples:
                log_reward.add(sample.logged_reward)
                weight = (
                    sample.target_propensities[sample.logged_action]
                    / sample.logged_propensities[sample.logged_action]
                )
                weight = self._weight_clamper(weight)
                rewards = input.target_model(sample.context)
                r1 = rewards[sample.logged_action]
                r2 = rewards[sample.target_action]
                tgt_reward.add((sample.logged_reward - r1) * weight + r2)
                rewards = input.ground_truth_model(sample.context)
                gt_reward.add(rewards[sample.target_action])
            self._append_estimate(
                log_reward.average, tgt_reward.average, gt_reward.average
            )
        return self.results
