#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
from reagent.evaluation.cpe import (
    CpeEstimate,
    CpeEstimateSet,
    bootstrapped_std_error_of_mean,
)
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
from reagent.ope.estimators.estimator import (
    Estimator,
    EstimatorResult,
    EstimatorResults,
)
from reagent.ope.estimators.sequential_estimators import (
    Action,
    ActionDistribution,
    DoublyRobustEstimator as SeqDREstimator,
    MAGICEstimator,
    RLEstimator,
    RLEstimatorInput,
)
from reagent.ope.estimators.types import ActionSpace


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OPEstimatorAdapter:
    def __init__(self, ope_estimator: Estimator, device=None):
        self._ope_estimator = ope_estimator
        self._device = device

    @staticmethod
    def edp_to_contextual_bandit_log(
        edp: EvaluationDataPage, device=None
    ) -> BanditsEstimatorInput:
        log = []
        n = edp.model_rewards.shape[0]
        for idx in range(n):
            # Action is only 1 if tgt policy and log policy took same action?
            action = torch.argmax(edp.action_mask[idx]).item()
            if edp.action_mask[idx][action] == 0.0:
                action = None
            logged_propensities = torch.zeros(
                edp.model_propensities[idx].shape, device=device
            )
            if action is not None:
                logged_propensities[action] = edp.logged_propensities[idx]
            log.append(
                LogSample(
                    context=None if edp.contexts is None else edp.contexts[idx],
                    log_action=Action(action),
                    log_reward=edp.logged_rewards[idx],
                    log_action_probabilities=ActionDistribution(logged_propensities),
                    tgt_action_probabilities=ActionDistribution(
                        edp.model_propensities[idx]
                    ),
                    tgt_action=Action(action),
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

    def estimate(self, edp: EvaluationDataPage, **kwargs) -> CpeEstimate:
        result = self._ope_estimator.evaluate(
            OPEstimatorAdapter.edp_to_contextual_bandit_log(edp), **kwargs
        )
        assert isinstance(result, EstimatorResult)
        logging.info(f"Got estimator result {result}, turning into cpe estimate")
        return OPEstimatorAdapter.estimator_result_to_cpe_estimate(result)


class SequentialOPEstimatorAdapter:
    def __init__(self, seq_ope_estimator: RLEstimator, gamma: float, device=None):
        self.seq_ope_estimator = seq_ope_estimator
        self.gamma = gamma
        self._device = device

    @staticmethod
    def estimator_results_to_cpe_estimate(
        estimator_results: EstimatorResults,
    ) -> CpeEstimate:
        scores = torch.tensor(
            [r.estimated_reward for r in estimator_results.results], dtype=torch.double
        )
        log_scores = torch.tensor(
            [r.log_reward for r in estimator_results.results], dtype=torch.double
        )

        dr_score = float(torch.mean(scores).item())
        dr_score_std_error = bootstrapped_std_error_of_mean(scores)

        log_score = float(torch.mean(log_scores).item())
        if log_score < 1e-6:
            logger.warning(
                "Can't normalize SDR-CPE because of small"
                f" or negative logged_policy_score ({log_score})."
                f"Episode values: {log_scores}."
            )
            return CpeEstimate(
                raw=dr_score,
                normalized=0.0,
                raw_std_error=dr_score_std_error,
                normalized_std_error=0.0,
            )
        return CpeEstimate(
            raw=dr_score,
            normalized=dr_score / log_score,
            raw_std_error=dr_score_std_error,
            normalized_std_error=dr_score_std_error / log_score,
        )

    def estimate(self, edp: EvaluationDataPage) -> CpeEstimate:
        est_input = edp.sequential_estimator_input
        assert est_input is not None, "EDP does not contain sequential estimator inputs"
        estimator_results = self.seq_ope_estimator.evaluate(
            RLEstimatorInput(
                gamma=self.gamma,
                log=est_input.log,
                target_policy=est_input.target_policy,
                value_function=est_input.value_function,
                discrete_states=est_input.discrete_states,
            )
        )
        assert isinstance(estimator_results, EstimatorResults)
        return SequentialOPEstimatorAdapter.estimator_results_to_cpe_estimate(
            estimator_results
        )


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

        self.ope_seq_dr_estimator = SequentialOPEstimatorAdapter(
            SeqDREstimator(device=self._device), gamma, device=self._device
        )
        self.ope_seq_weighted_dr_estimator = SequentialOPEstimatorAdapter(
            SeqDREstimator(weighted=True, device=self._device),
            gamma,
            device=self._device,
        )
        self.ope_seq_magic_estimator = SequentialOPEstimatorAdapter(
            MAGICEstimator(device=self._device), gamma
        )

    def score_cpe(self, metric_name, edp: EvaluationDataPage):
        logger.info("Using OPE adapter")
        direct_method = self.ope_dm_estimator.estimate(edp)
        inverse_propensity = self.ope_ips_estimator.estimate(edp)
        doubly_robust = self.ope_dr_estimator.estimate(edp)

        sequential_doubly_robust = self.ope_seq_dr_estimator.estimate(edp)
        weighted_doubly_robust = self.ope_seq_weighted_dr_estimator.estimate(edp)
        magic = self.ope_seq_magic_estimator.estimate(edp)
        return CpeEstimateSet(
            direct_method=direct_method,
            inverse_propensity=inverse_propensity,
            doubly_robust=doubly_robust,
            sequential_doubly_robust=sequential_doubly_robust,
            weighted_doubly_robust=weighted_doubly_robust,
            magic=magic,
        )
