#!/usr/bin/env python3

from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.evaluation.cpe import CpeEstimate
from reagent.workflow.result_registries import TrainingReport


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


@dataclass
class ActorCriticTrainingReport(TrainingReport):
    __registry_name__ = "actor_critic_report"


@dataclass
class ParametricDQNTrainingReport(TrainingReport):
    __registry_name__ = "parametric_dqn_report"
