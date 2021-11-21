#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from reagent.evaluation.cpe import CpeEstimate, bootstrapped_std_error_of_mean
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from torch import Tensor


logger = logging.getLogger(__name__)


DEFAULT_FRAC_TRAIN = 0.4
DEFAULT_FRAC_VALID = 0.1


class DoublyRobustHP(NamedTuple):
    frac_train: float = DEFAULT_FRAC_TRAIN
    frac_valid: float = DEFAULT_FRAC_VALID
    bootstrap_num_samples: int = 1000
    bootstrap_sample_percent: float = 0.25
    xgb_params: Optional[Dict[str, Union[float, int, str]]] = None
    bope_mode: Optional[str] = None
    bope_num_samples: Optional[int] = None


class TrainValidEvalData(NamedTuple):
    contexts_dict: Dict[str, Tensor]
    model_propensities_dict: Dict[str, Tensor]
    actions_logged_dict: Dict[str, Tensor]
    action_mask_dict: Dict[str, Tensor]
    logged_rewards_dict: Dict[str, Tensor]
    model_rewards_dict: Dict[str, Tensor]
    model_rewards_for_logged_action_dict: Dict[str, Tensor]
    logged_propensities_dict: Dict[str, Tensor]
    num_examples_dict: Dict[str, int]


@dataclass
class EstimationData:
    contexts_actions_train: Optional[Tensor]
    policy_indicators_train: Optional[Tensor]
    weights_train: Optional[Tensor]
    contexts_actions_valid: Optional[Tensor]
    policy_indicators_valid: Optional[Tensor]
    weights_valid: Optional[Tensor]
    contexts_actions_eval: Optional[Tensor]
    contexts_train: Optional[Tensor]
    actions_logged_train: Optional[Tensor]
    contexts_valid: Optional[Tensor]
    actions_logged_valid: Optional[Tensor]
    contexts_eval: Optional[Tensor]
    actions_logged_eval: Optional[Tensor]
    model_propensities_eval: Tensor
    model_rewards_eval: Tensor
    action_mask_eval: Tensor
    logged_rewards_eval: Tensor
    model_rewards_for_logged_action_eval: Tensor
    logged_propensities_eval: Tensor

    def __post_init__(self):
        assert (
            self.model_propensities_eval.shape
            == self.model_rewards_eval.shape
            == self.action_mask_eval.shape
        ) and len(self.model_propensities_eval.shape) == 2, (
            f"{self.model_propensities_eval.shape} "
            f"{self.model_rewards_eval.shape} "
            f"{self.action_mask_eval.shape}"
        )
        assert (
            (
                self.logged_rewards_eval.shape
                == self.model_rewards_for_logged_action_eval.shape
                == self.logged_propensities_eval.shape
            )
            and len(self.logged_rewards_eval.shape) == 2
            and self.logged_rewards_eval.shape[1] == 1
        ), (
            f"{self.logged_rewards_eval.shape} "
            f"{self.model_rewards_for_logged_action_eval.shape} "
            f"{self.logged_propensities_eval.shape}"
        )


class ImportanceSamplingData(NamedTuple):
    importance_weight: Tensor
    logged_rewards: Tensor
    model_rewards: Tensor
    model_rewards_for_logged_action: Tensor
    model_propensities: Tensor


class DoublyRobustEstimator:
    """
    For details, visit https://arxiv.org/pdf/1612.01205.pdf
    """

    def _split_data(
        self,
        edp: EvaluationDataPage,
        frac_train: float = DEFAULT_FRAC_TRAIN,
        frac_valid: float = DEFAULT_FRAC_VALID,
    ) -> TrainValidEvalData:
        """
        Split the data into training, validation and evaluation parts.
        Training and validation and used for model training to estimate
        the importance weights.
        Only evaluation data is used for policy estimation.

        This function is used for BOP-E and Estimated Propensity Score methods,
        but not for the standard Doubly Robust estimator.
        """
        num_examples = edp.model_propensities.shape[0]
        # split data into training, validation and eval
        indices = np.random.permutation(num_examples)
        idx_train = indices[0 : int(frac_train * num_examples)]
        idx_valid = indices[
            int(frac_train * num_examples) : int(
                (frac_train + frac_valid) * num_examples
            )
        ]
        idx_eval = indices[int((frac_train + frac_valid) * num_examples) :]
        if edp.contexts is None:
            raise ValueError("contexts not provided in input")
        contexts_dict = {
            "train": edp.contexts[idx_train],
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            "valid": edp.contexts[idx_valid],
            "eval": edp.contexts[idx_eval],
        }
        model_propensities_dict = {
            "train": edp.model_propensities[idx_train],
            "valid": edp.model_propensities[idx_valid],
            "eval": edp.model_propensities[idx_eval],
        }
        # edp.action_mask is N*N_actions tensor of indicators of which actions
        # were actually taken by the logged algo
        actions_logged = torch.max(edp.action_mask, dim=1, keepdim=True)[1].float()
        actions_logged_dict = {
            "train": actions_logged[idx_train],
            "valid": actions_logged[idx_valid],
            "eval": actions_logged[idx_eval],
        }
        action_mask_dict = {
            "train": edp.action_mask[idx_train],
            "valid": edp.action_mask[idx_valid],
            "eval": edp.action_mask[idx_eval],
        }
        logged_rewards_dict = {
            "train": edp.logged_rewards[idx_train],
            "valid": edp.logged_rewards[idx_valid],
            "eval": edp.logged_rewards[idx_eval],
        }
        model_rewards_dict = {
            "train": edp.model_rewards[idx_train],
            "valid": edp.model_rewards[idx_valid],
            "eval": edp.model_rewards[idx_eval],
        }
        model_rewards_for_logged_action_dict = {
            "train": edp.model_rewards_for_logged_action[idx_train],
            "valid": edp.model_rewards_for_logged_action[idx_valid],
            "eval": edp.model_rewards_for_logged_action[idx_eval],
        }
        logged_propensities_dict = {
            "train": edp.logged_propensities[idx_train],
            "valid": edp.logged_propensities[idx_valid],
            "eval": edp.logged_propensities[idx_eval],
        }
        num_examples_dict = {
            "train": int(frac_train * num_examples),
            "valid": int((frac_train + frac_valid) * num_examples)
            - int(frac_train * num_examples),
            "eval": num_examples - int((frac_train + frac_valid) * num_examples),
        }
        return TrainValidEvalData(
            contexts_dict=contexts_dict,
            model_propensities_dict=model_propensities_dict,
            actions_logged_dict=actions_logged_dict,
            action_mask_dict=action_mask_dict,
            logged_rewards_dict=logged_rewards_dict,
            model_rewards_dict=model_rewards_dict,
            model_rewards_for_logged_action_dict=model_rewards_for_logged_action_dict,
            logged_propensities_dict=logged_propensities_dict,
            num_examples_dict=num_examples_dict,
        )

    def _prepare_data(self, edp: EvaluationDataPage) -> EstimationData:
        ed = EstimationData(
            contexts_actions_train=None,
            policy_indicators_train=None,
            weights_train=None,
            contexts_actions_valid=None,
            policy_indicators_valid=None,
            weights_valid=None,
            contexts_actions_eval=None,
            contexts_train=None,
            actions_logged_train=None,
            contexts_valid=None,
            actions_logged_valid=None,
            contexts_eval=edp.contexts,
            actions_logged_eval=None,
            model_propensities_eval=edp.model_propensities,
            model_rewards_eval=edp.model_rewards,
            action_mask_eval=edp.action_mask,
            logged_rewards_eval=edp.logged_rewards,
            model_rewards_for_logged_action_eval=edp.model_rewards_for_logged_action,
            logged_propensities_eval=edp.logged_propensities,
        )
        return ed

    def _get_importance_sampling_inputs(
        self, ed: EstimationData
    ) -> ImportanceSamplingData:
        target_propensity_for_action = torch.sum(
            ed.model_propensities_eval * ed.action_mask_eval, dim=1, keepdim=True
        )
        # target_propensity_for_action is N*1 tensor of target algo propensities
        # for historical actions
        # logged_propensities_eval is N*1 tensor of propensity scores for historical
        # actions by the prod algorithm at each context
        importance_weights = (
            target_propensity_for_action / ed.logged_propensities_eval
        ).float()
        logger.info(f"Mean IPS weight on the eval dataset: {importance_weights.mean()}")
        return ImportanceSamplingData(
            importance_weight=importance_weights,
            logged_rewards=ed.logged_rewards_eval,
            model_rewards=ed.model_rewards_eval,
            model_rewards_for_logged_action=ed.model_rewards_for_logged_action_eval,
            model_propensities=ed.model_propensities_eval,
        )

    def _get_importance_sampling_estimates(
        self, isd: ImportanceSamplingData, hp: DoublyRobustHP
    ) -> Tuple[CpeEstimate, CpeEstimate, CpeEstimate]:
        # The score we would get if we evaluate the logged policy against itself
        logged_policy_score = float(
            torch.mean(isd.logged_rewards)
        )  # logged_rewards is N*1 tensor of historical rewards
        if logged_policy_score < 1e-6:
            logger.warning(
                "Can't normalize DR-CPE because of small or negative "
                + "logged_policy_score"
            )
            normalizer = 0.0
        else:
            normalizer = 1.0 / logged_policy_score

        if isd.model_rewards is None:
            # Fill with zero, equivalent to just doing IPS
            direct_method_values = torch.zeros(
                [isd.model_propensities.shape[0], 1], dtype=torch.float32
            )
        else:
            # model rewards is (N_samples)*N_actions tensor of predicted
            # counterfactual rewards for each possible action at each
            # historical context
            direct_method_values = torch.sum(
                isd.model_propensities * isd.model_rewards, dim=1, keepdim=True
            )

        direct_method_score = float(torch.mean(direct_method_values))
        logger.info(
            f"Normalized Direct method score = {direct_method_score * normalizer}"
        )
        direct_method_std_error = bootstrapped_std_error_of_mean(
            direct_method_values.squeeze(),
            sample_percent=hp.bootstrap_sample_percent,
            num_samples=hp.bootstrap_num_samples,
        )
        direct_method_estimate = CpeEstimate(
            raw=direct_method_score,
            normalized=direct_method_score * normalizer,
            raw_std_error=direct_method_std_error,
            normalized_std_error=direct_method_std_error * normalizer,
        )

        ips = isd.importance_weight * isd.logged_rewards  # N*1

        doubly_robust = (
            isd.importance_weight
            * (isd.logged_rewards - isd.model_rewards_for_logged_action)
        ) + direct_method_values
        # model_rewards_for_logged_action is N*1 of estimated rewards for target
        # policy

        ips_score = float(torch.mean(ips))
        logger.info(f"Normalized IPS score = {ips_score * normalizer}")

        ips_score_std_error = bootstrapped_std_error_of_mean(
            ips.squeeze(),
            sample_percent=hp.bootstrap_sample_percent,
            num_samples=hp.bootstrap_num_samples,
        )
        inverse_propensity_estimate = CpeEstimate(
            raw=ips_score,
            normalized=ips_score * normalizer,
            raw_std_error=ips_score_std_error,
            normalized_std_error=ips_score_std_error * normalizer,
        )

        dr_score = float(torch.mean(doubly_robust))
        dr_score_std_error = bootstrapped_std_error_of_mean(
            doubly_robust.squeeze(),
            sample_percent=hp.bootstrap_sample_percent,
            num_samples=hp.bootstrap_num_samples,
        )
        doubly_robust_estimate = CpeEstimate(
            raw=dr_score,
            normalized=dr_score * normalizer,
            raw_std_error=dr_score_std_error,
            normalized_std_error=dr_score_std_error * normalizer,
        )

        return (
            direct_method_estimate,
            inverse_propensity_estimate,
            doubly_robust_estimate,
        )

    def estimate(
        self, edp: EvaluationDataPage, hp: Optional[DoublyRobustHP] = None
    ) -> Tuple[CpeEstimate, CpeEstimate, CpeEstimate]:
        hp = hp or DoublyRobustHP()
        ed = self._prepare_data(edp)
        isd = self._get_importance_sampling_inputs(ed)
        return self._get_importance_sampling_estimates(isd, hp=hp)
