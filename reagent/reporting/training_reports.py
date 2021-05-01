#!/usr/bin/env python3

from typing import Dict, Optional

# @manual=third-party//pandas:pandas-py
from pandas import DataFrame
from reagent.core.dataclasses import dataclass
from reagent.core.result_registries import TrainingReport
from reagent.evaluation.cpe import CpeEstimate


@dataclass
class DQNTrainingReport(TrainingReport):
    __registry_name__ = "dqn_report"

    td_loss: Optional[float] = None
    mc_loss: Optional[float] = None
    reward_ips: Optional[CpeEstimate] = None
    reward_dm: Optional[CpeEstimate] = None
    reward_dr: Optional[CpeEstimate] = None
    value_sequential_dr: Optional[CpeEstimate] = None
    value_weighted_dr: Optional[CpeEstimate] = None
    value_magic_dr: Optional[CpeEstimate] = None
    cpe_metrics_table: Optional[DataFrame] = None
    logged_action_distribution: Optional[Dict[str, float]] = None
    model_action_distribution: Optional[Dict[str, float]] = None
    model_logged_dist_kl_divergence: Optional[float] = None
    target_distribution_error: Optional[float] = None
    q_value_means: Optional[Dict[str, float]] = None
    eval_q_value_means: Optional[Dict[str, float]] = None
    eval_q_value_stds: Optional[Dict[str, float]] = None
    eval_action_distribution: Optional[Dict[str, float]] = None
    q_value_kl_divergence: Optional[float] = None
    action_dist_kl_divergence: Optional[float] = None
