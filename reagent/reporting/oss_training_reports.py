#!/usr/bin/env python3

from typing import List, Optional

from reagent.core.dataclasses import dataclass
from reagent.evaluation.cpe import CpeEstimate
from reagent.reporting.training_reports import TrainingReport


@dataclass
class OssDQNTrainingReport(TrainingReport):
    __registry_name__ = "oss_dqn_report"

    td_loss: Optional[float] = None
    mc_loss: Optional[float] = None
    reward_ips: Optional[CpeEstimate] = None
    reward_dm: Optional[CpeEstimate] = None
    reward_dr: Optional[CpeEstimate] = None
    value_sequential_dr: Optional[CpeEstimate] = None
    value_weighted_dr: Optional[CpeEstimate] = None
    value_magic_dr: Optional[CpeEstimate] = None


@dataclass
class OssActorCriticTrainingReport(TrainingReport):
    __registry_name__ = "oss_actor_critic_report"


@dataclass
class OssParametricDQNTrainingReport(TrainingReport):
    __registry_name__ = "oss_parametric_dqn_report"

    td_loss: Optional[float] = None
    mc_loss: Optional[float] = None
    reward_ips: Optional[CpeEstimate] = None
    reward_dm: Optional[CpeEstimate] = None
    reward_dr: Optional[CpeEstimate] = None
    value_sequential_dr: Optional[CpeEstimate] = None
    value_weighted_dr: Optional[CpeEstimate] = None
    value_magic_dr: Optional[CpeEstimate] = None


@dataclass
class OssWorldModelTrainingReport(TrainingReport):
    __registry_name__ = "oss_world_model_report"
    loss: List[float]
    gmm: List[float]
    bce: List[float]
    mse: List[float]


@dataclass
class DebugToolsReport(TrainingReport):
    __registry_name__ = "oss_debug_tools_report"

    feature_importance: Optional[List[float]] = None
    feature_sensitivity: Optional[List[float]] = None


@dataclass
class OssRankingModelTrainingReport(TrainingReport):
    __registry_name__ = "oss_ranking_model_training_report"
