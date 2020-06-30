#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch
from reagent.evaluation.cpe import CpeEstimate, CpeEstimateSet
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import Evaluator
from reagent.ope.estimators.contextual_bandits_estimators import (
    BanditsEstimatorInput,
    DMEstimator,
    DoublyRobustEstimator,
    IPSEstimator,
    LogSample,
    ModelOutputs,
)
from reagent.ope.estimators.estimator import Estimator, EstimatorResult
from reagent.ope.estimators.types import ActionSpace


class OPEstimatorAdapter:
    def __init__(self, ope_estimator: Estimator):
        self._ope_estimator = ope_estimator

    @staticmethod
    def edp_to_contextual_bandit_log(edp: EvaluationDataPage) -> BanditsEstimatorInput:
        log = []
        n = edp.model_rewards.shape[0]
        for idx in range(n):
            # Action is only 1 if tgt policy and log policy took same action?
            action = torch.argmax(edp.action_mask[idx]).item()
            if edp.action_mask[idx][action] == 0.0:
                action = None
            logged_propensities = torch.zeros(edp.model_propensities[idx].shape)
            if action is not None:
                logged_propensities[action] = edp.logged_propensities[idx]
            log.append(
                LogSample(
                    context=None if edp.contexts is None else edp.contexts[idx],
                    log_action=action,
                    log_reward=edp.logged_rewards[idx],
                    log_action_probabilities=logged_propensities,
                    tgt_action_probabilities=edp.model_propensities[idx],
                    tgt_action=action,
                    model_outputs=ModelOutputs(
                        tgt_reward_from_log_action=edp.model_rewards_for_logged_action[
                            idx
                        ],
                        tgt_rewards=edp.model_rewards[idx],
                    )
                    # item features not specified as edp came from trained reward model
                )
            )
        return BanditsEstimatorInput(ActionSpace(edp.action_mask.shape[1]), log, True)

    @staticmethod
    def estimator_result_to_cpe_estimate(result: EstimatorResult) -> CpeEstimate:
        assert result.estimated_reward_normalized is not None
        assert result.estimated_reward_normalized is not None
        assert result.estimated_reward_std_error is not None
        assert result.estimated_reward_normalized_std_error is not None
        return CpeEstimate(
            raw=result.estimated_reward,
            normalized=result.estimated_reward_normalized,
            raw_std_error=result.estimated_reward_std_error,
            normalized_std_error=result.estimated_reward_normalized_std_error,
        )

    def estimate(self, edp: EvaluationDataPage) -> CpeEstimate:
        result = self._ope_estimator.evaluate(
            OPEstimatorAdapter.edp_to_contextual_bandit_log(edp)
        )
        assert isinstance(result, EstimatorResult)
        return OPEstimatorAdapter.estimator_result_to_cpe_estimate(result)


class OPEvaluator(Evaluator):
    def __init__(
        self, action_names, gamma, model, metrics_to_score=None, device=None
    ) -> None:
        super().__init__(action_names, gamma, model, metrics_to_score)

        self._device = device
        self.ope_dm_estimator = OPEstimatorAdapter(DMEstimator(device=self._device))
        self.ope_ips_estimator = OPEstimatorAdapter(IPSEstimator(device=self._device))
        self.ope_dr_estimator = OPEstimatorAdapter(
            DoublyRobustEstimator(device=self._device)
        )

    def score_cpe(self, metric_name, edp: EvaluationDataPage):
        direct_method = self.ope_dm_estimator.estimate(edp)
        inverse_propensity = self.ope_ips_estimator.estimate(edp)
        doubly_robust = self.ope_dr_estimator.estimate(edp)

        sequential_doubly_robust = self.sequential_doubly_robust_estimator.estimate(edp)
        weighted_doubly_robust = self.weighted_sequential_doubly_robust_estimator.estimate(
            edp, num_j_steps=1, whether_self_normalize_importance_weights=True
        )
        magic = self.weighted_sequential_doubly_robust_estimator.estimate(
            edp,
            num_j_steps=Evaluator.NUM_J_STEPS_FOR_MAGIC_ESTIMATOR,
            whether_self_normalize_importance_weights=True,
        )
        return CpeEstimateSet(
            direct_method=direct_method,
            inverse_propensity=inverse_propensity,
            doubly_robust=doubly_robust,
            sequential_doubly_robust=sequential_doubly_robust,
            weighted_doubly_robust=weighted_doubly_robust,
            magic=magic,
        )
