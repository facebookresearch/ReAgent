#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import xgboost as xgb
from reagent.evaluation.cpe import CpeEstimate, bootstrapped_std_error_of_mean
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from torch import Tensor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class DoublyRobustEstimatorBOPE(DoublyRobustEstimator):
    """
    This class implements a doubly-robust Balanced Off-Policy Evaluation (BOP-E)
    method.
    For details about BOP-E see https://arxiv.org/abs/1906.03694
    For analysis of BOP-E performance see https://fburl.com/bope_eval_nb

    Note that when using BOP-E the data gets split into training, validation
    and evaluation parts and only the evaluation part is used directly for policy
    evaluation, while training and validation datasets are used for model training.

    supported modes (all doubly robust):
    1. bope_weights. Use BOP-E (ignoring logged propensities) to estimate the
        importance weights. Propensities of the target policy are used as
        observation weights when training BOP-E classifier.
    2. bope_weighted_targets. Use BOP-E (ignoring logged propensities) to
        estimate the importance weights. Propensities of the target policy
        are used as soft targets to train BOP-E regressor. With this method
        BOP-E trains a regressor instead of a classifier.
    3. bope_sampling. Use BOP-E (ignoring logged propensities)
        to estimate the importance weights. Propensities of the target policy
        are used to sample the actions for the classifier training data.
    """

    def _prepare_data(self, edp: EvaluationDataPage) -> EstimationData:
        """
        Prepare the datasets for BOP-E classifier estimation
        """
        assert (
            edp.contexts is not None
        ), "edp.contexts have to be specified when using the estimation-based methods"
        num_actions = edp.model_propensities.shape[1]
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `frac_train`.
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `frac_valid`.
        tved = self._split_data(edp, self.frac_train, self.frac_valid)

        actions_target_dict = {}
        contexts_actions_target_dict = {}
        weights_target_dict = {}
        policy_indicators_target_dict = {}
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `mode`.
        if self.mode == "bope_sampling":
            for d in ["train", "valid"]:
                # model_propensities is N*N_actions tensor of propensity scores
                # for each possible action by the target algorithm at each context
                actions_target_dict[d] = (
                    torch.multinomial(
                        tved.model_propensities_dict[d],
                        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no
                        #  attribute `num_samples`.
                        self.num_samples,
                        replacement=True,
                    )
                    .float()
                    .transpose(0, 1)
                    .contiguous()
                    .view(-1, 1)
                )
                # transpose and reshape so that the contexts (rows) are arranged
                # like [C1,...,CN,C1,...,CN,.....,C1,...,CN]

                # TODO: add context-action interaction here
                contexts_actions_target_dict[d] = torch.cat(
                    [
                        torch.cat([tved.contexts_dict[d]] * self.num_samples, dim=0),
                        actions_target_dict[d],
                    ],
                    dim=1,
                )
                weights_target_dict[d] = torch.ones(
                    tved.num_examples_dict[d] * self.num_samples, 1
                )
                policy_indicators_target_dict[d] = torch.ones(
                    tved.num_examples_dict[d] * self.num_samples, 1, dtype=torch.float32
                )
        elif self.mode == "bope_weights":
            # rows are outer products of actions and contexts, ordered first by
            # context and then by action
            # [[C0,A0], [C0,A1], [C0,A2], [C1,A0], [C1,A1], [C1,A2],...]
            for d in ["train", "valid"]:
                actions_target_dict[d] = torch.tensor(
                    list(
                        itertools.chain.from_iterable(
                            [
                                [x] * tved.num_examples_dict[d]
                                for x in range(num_actions)
                            ]
                        )
                    ),
                    dtype=torch.float32,
                ).view(-1, 1)
                weights_target_dict[d] = (
                    tved.model_propensities_dict[d]
                    .transpose(0, 1)
                    .contiguous()
                    .view(-1, 1)
                )
                policy_indicators_target_dict[d] = torch.ones(
                    tved.num_examples_dict[d] * num_actions, 1, dtype=torch.float32
                )
                # TODO: add context-action interaction here
                contexts_actions_target_dict[d] = torch.cat(
                    [
                        torch.cat([tved.contexts_dict[d]] * num_actions, dim=0),  # 1498
                        actions_target_dict[d],  # 1496
                    ],
                    dim=1,
                )
        elif self.mode == "bope_weighted_targets":
            # rows are outer products of actions and contexts, ordered first by
            # context and then by action
            # [[C0,A0], [C0,A1], [C0,A2], [C1,A0], [C1,A1], [C1,A2],...]
            for d in ["train", "valid"]:
                actions_target_dict[d] = torch.tensor(
                    list(
                        itertools.chain.from_iterable(
                            [
                                [x] * tved.num_examples_dict[d]
                                for x in range(num_actions)
                            ]
                        )
                    ),
                    dtype=torch.float32,
                ).view(-1, 1)
                weights_target_dict[d] = torch.ones(
                    tved.num_examples_dict[d] * num_actions, 1
                )
                policy_indicators_target_dict[d] = (
                    tved.model_propensities_dict[d]
                    .transpose(0, 1)
                    .contiguous()
                    .view(-1, 1)
                )
                # TODO: add context-action interaction here
                contexts_actions_target_dict[d] = torch.cat(
                    [
                        torch.cat([tved.contexts_dict[d]] * num_actions, dim=0),
                        actions_target_dict[d],
                    ],
                    dim=1,
                )
        else:
            raise ValueError("BOP-E mode '{}'' not supported".format(self.mode))
        contexts_actions_logged_dict = {}
        weights_logged_dict = {}
        policy_indicators_logged_dict = {}
        contexts_actions_all_dict = {}
        policy_indicators_all_dict = {}
        weights_all_dict = {}
        for d in ["train", "valid"]:
            contexts_actions_logged_dict[d] = torch.cat(
                (tved.contexts_dict[d], tved.actions_logged_dict[d]), dim=1
            )  # N*(d+1)
            weights_logged_dict[d] = torch.ones(
                tved.num_examples_dict[d], 1, dtype=torch.float32
            )
            policy_indicators_logged_dict[d] = torch.zeros(
                tved.num_examples_dict[d], 1, dtype=torch.float32
            )
            contexts_actions_all_dict[d] = torch.cat(
                [contexts_actions_logged_dict[d], contexts_actions_target_dict[d]],
                dim=0,
            ).numpy()
            policy_indicators_all_dict[d] = torch.cat(
                [policy_indicators_logged_dict[d], policy_indicators_target_dict[d]],
                dim=0,
            ).numpy()
            weights_all_dict[d] = (
                torch.cat([weights_logged_dict[d], weights_target_dict[d]], dim=0)
                .flatten()
                .numpy()
            )
            if (
                contexts_actions_all_dict[d].shape[0]
                != policy_indicators_all_dict[d].shape[0]
            ):
                raise ValueError(
                    "number of rows in {} contexts_actions({}) and policy_"
                    "indicators({}) has to be equal".format(
                        d,
                        contexts_actions_all_dict[d].shape[0],
                        policy_indicators_all_dict[d].shape[0],
                    )
                )
            if contexts_actions_all_dict[d].shape[0] != weights_all_dict[d].shape[0]:
                raise ValueError(
                    "number of rows in {} contexts_actions({}) and weights({})"
                    " has to be equal".format(
                        d,
                        contexts_actions_all_dict[d].shape[0],
                        weights_all_dict[d].shape[0],
                    )
                )
        contexts_actions_logged_dict["eval"] = torch.cat(
            (tved.contexts_dict["eval"], tved.actions_logged_dict["eval"]), dim=1
        )  # N*(d+1)

        return EstimationData(
            contexts_actions_train=contexts_actions_all_dict["train"],
            policy_indicators_train=policy_indicators_all_dict["train"],
            weights_train=weights_all_dict["train"],
            contexts_actions_valid=contexts_actions_all_dict["valid"],
            policy_indicators_valid=policy_indicators_all_dict["valid"],
            weights_valid=weights_all_dict["valid"],
            contexts_actions_eval=contexts_actions_logged_dict["eval"],
            contexts_train=None,
            actions_logged_train=None,
            contexts_valid=None,
            actions_logged_valid=None,
            contexts_eval=None,
            actions_logged_eval=None,
            model_propensities_eval=tved.model_propensities_dict["eval"],
            model_rewards_eval=tved.model_rewards_dict["eval"],
            action_mask_eval=tved.action_mask_dict["eval"],
            logged_rewards_eval=tved.logged_rewards_dict["eval"],
            model_rewards_for_logged_action_eval=tved.model_rewards_for_logged_action_dict[
                "eval"
            ],
            logged_propensities_eval=tved.logged_propensities_dict["eval"],
        )

    def _estimate_xgboost_model(
        self,
        ed: EstimationData,
        # pyre-fixme[9]: xgb_params has type `Dict[str, Union[float, int, str]]`;
        #  used as `None`.
        xgb_params: Dict[str, Union[str, float, int]] = None,
        nthread: int = 8,
    ) -> xgb.Booster:
        if xgb_params is None:
            xgb_params = {}
        dmatrix_train = xgb.DMatrix(
            ed.contexts_actions_train,
            ed.policy_indicators_train,
            nthread=nthread,
            weight=ed.weights_train,
        )
        dmatrix_valid = xgb.DMatrix(
            ed.contexts_actions_valid,
            ed.policy_indicators_valid,
            nthread=nthread,
            weight=ed.weights_valid,
        )
        if xgb_params is not None:  # check for None to satisfy a test
            xgb_params.update({"objective": "binary:logistic"})
        classifier: xgb.Booster = xgb.train(
            xgb_params,
            dmatrix_train,
            evals=[(dmatrix_valid, "validation_set")],
            verbose_eval=False,
        )
        return classifier

    def _get_importance_sampling_inputs(
        self,
        ed: EstimationData,
        # pyre-fixme[9]: xgb_params has type `Dict[str, Union[float, int, str]]`;
        #  used as `None`.
        xgb_params: Dict[str, Union[str, float, int]] = None,
    ) -> ImportanceSamplingData:
        classifier = self._estimate_xgboost_model(ed, xgb_params)

        # predictions are made only for the eval set to prevent classifier
        # overfitting
        predictions = classifier.predict(xgb.DMatrix(ed.contexts_actions_eval))

        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `mode`.
        if self.mode == "bope_sampling":
            # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute
            #  `num_samples`.
            bope_weight_normalization_factor = 1.0 / self.num_samples
        else:
            bope_weight_normalization_factor = 1.0

        importance_weights = (
            torch.tensor(predictions / (1.0 - predictions), dtype=torch.float32).view(
                -1, 1
            )
            * bope_weight_normalization_factor
        )
        return ImportanceSamplingData(
            importance_weight=importance_weights,
            logged_rewards=ed.logged_rewards_eval,
            model_rewards=ed.model_rewards_eval,
            model_rewards_for_logged_action=ed.model_rewards_for_logged_action_eval,
            model_propensities=ed.model_propensities_eval,
        )

    def estimate(
        self, edp: EvaluationDataPage, hp: Optional[DoublyRobustHP] = None
    ) -> Tuple[CpeEstimate, CpeEstimate, CpeEstimate]:
        if hp is None:
            raise ValueError("Hyperparameters have to be provided for BOP-E")
        if hp.bope_mode is None:
            raise ValueError("bope_mode has to be specified in hyperparameters")
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `mode`.
        self.mode = hp.bope_mode
        if (self.mode == "bope_sampling") and (hp.bope_num_samples is None):
            raise ValueError(
                "Number of samples has to be specified for mode 'bope_sampling'"
            )
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `num_samples`.
        self.num_samples = 0 if hp.bope_num_samples is None else hp.bope_num_samples
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `frac_train`.
        self.frac_train = hp.frac_train
        # pyre-fixme[16]: `DoublyRobustEstimatorBOPE` has no attribute `frac_valid`.
        self.frac_valid = hp.frac_train
        xgb_params: Dict[str, Union[str, float, int]] = hp.xgb_params or {}
        ed = self._prepare_data(edp)
        isd = self._get_importance_sampling_inputs(ed, xgb_params)
        return self._get_importance_sampling_estimates(isd, hp=hp)


class DoublyRobustEstimatorEstProp(DoublyRobustEstimator):
    def _prepare_data(self, edp: EvaluationDataPage) -> EstimationData:
        assert (
            edp.contexts is not None
        ), "edp.contexts have to be specified when using the estimation-based methods"
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `num_actions`.
        self.num_actions = edp.model_propensities.shape[1]
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `frac_train`.
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `frac_valid`.
        tved = self._split_data(edp, self.frac_train, self.frac_valid)

        return EstimationData(
            contexts_actions_train=None,
            policy_indicators_train=None,
            weights_train=None,
            contexts_actions_valid=None,
            policy_indicators_valid=None,
            weights_valid=None,
            contexts_actions_eval=None,
            contexts_train=tved.contexts_dict["train"],
            actions_logged_train=tved.actions_logged_dict["train"],
            contexts_valid=tved.contexts_dict["valid"],
            actions_logged_valid=tved.actions_logged_dict["valid"],
            contexts_eval=tved.contexts_dict["eval"],
            actions_logged_eval=tved.actions_logged_dict["eval"],
            model_propensities_eval=tved.model_propensities_dict["eval"],
            model_rewards_eval=tved.model_rewards_dict["eval"],
            action_mask_eval=tved.action_mask_dict["eval"],
            logged_rewards_eval=tved.logged_rewards_dict["eval"],
            model_rewards_for_logged_action_eval=tved.model_rewards_for_logged_action_dict[
                "eval"
            ],
            logged_propensities_eval=tved.logged_propensities_dict["eval"],
        )

    def _estimate_xgboost_model(
        self,
        ed: EstimationData,
        num_classes: int,
        # pyre-fixme[9]: xgb_params has type `Dict[str, Union[float, int, str]]`;
        #  used as `None`.
        xgb_params: Dict[str, Union[str, float, int]] = None,
        nthread: int = 8,
    ) -> xgb.Booster:
        if xgb_params is None:
            xgb_params = {}
        dmatrix_train = xgb.DMatrix(
            ed.contexts_train, ed.actions_logged_train, nthread=nthread
        )
        dmatrix_valid = xgb.DMatrix(
            ed.contexts_valid, ed.actions_logged_valid, nthread=nthread
        )
        xgb_params = xgb_params.copy()
        xgb_params.update(
            {"objective": "multi:softprob", "num_class": num_classes, "n_gpus": 0}
        )
        classifier: xgb.Booster = xgb.train(
            xgb_params,
            dmatrix_train,
            evals=[(dmatrix_valid, "validation_set")],
            verbose_eval=False,
        )
        return classifier

    def _get_importance_sampling_inputs(
        self,
        ed: EstimationData,
        # pyre-fixme[9]: xgb_params has type `Dict[str, Union[float, int, str]]`;
        #  used as `None`.
        xgb_params: Dict[str, Union[str, float, int]] = None,
    ):
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `num_actions`.
        classifier = self._estimate_xgboost_model(ed, self.num_actions, xgb_params)
        # predictions are made only for the eval set to prevent classifier
        # overfitting
        predicted_logged_propensities_all_actions = torch.tensor(
            classifier.predict(xgb.DMatrix(ed.contexts_eval)), dtype=torch.float32
        )
        if ed.actions_logged_eval is None:
            raise ValueError("ed.actions_logged_eval has to be non-None")
        ret = predicted_logged_propensities_all_actions.gather(
            1,
            # pyre-fixme[16]: `Optional` has no attribute `long`.
            ed.actions_logged_eval.long(),
        )
        predicted_logged_policy_propensities_logged_actions = ret

        target_propensity_for_action = torch.sum(
            ed.model_propensities_eval * ed.action_mask_eval, dim=1, keepdim=True
        )

        importance_weights = (
            target_propensity_for_action
            / predicted_logged_policy_propensities_logged_actions
        ).float()
        return ImportanceSamplingData(
            importance_weight=importance_weights,
            logged_rewards=ed.logged_rewards_eval,
            model_rewards=ed.model_rewards_eval,
            model_rewards_for_logged_action=ed.model_rewards_for_logged_action_eval,
            model_propensities=ed.model_propensities_eval,
        )

    def estimate(
        self, edp: EvaluationDataPage, hp: Optional[DoublyRobustHP] = None
    ) -> Tuple[CpeEstimate, CpeEstimate, CpeEstimate]:
        hp = hp or DoublyRobustHP()
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `frac_train`.
        self.frac_train = hp.frac_train
        # pyre-fixme[16]: `DoublyRobustEstimatorEstProp` has no attribute `frac_valid`.
        self.frac_valid = hp.frac_valid
        xgb_params = hp.xgb_params or {}
        ed = self._prepare_data(edp)
        isd = self._get_importance_sampling_inputs(ed, xgb_params)
        return self._get_importance_sampling_estimates(isd, hp=hp)
