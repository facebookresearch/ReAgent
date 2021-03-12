#!/usr/bin/env python3

from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_registries import TrainingReport
from reagent.evaluation.cpe import CpeEstimate


@dataclass
class ActorCriticTrainingReport(TrainingReport):
    __registry_name__ = "actor_critic_report"


@dataclass
class ParametricDQNTrainingReport(TrainingReport):
    __registry_name__ = "parametric_dqn_report"


@dataclass
class SlateQTrainingReport(TrainingReport):
    __registry_name__ = "slate_q_report"
