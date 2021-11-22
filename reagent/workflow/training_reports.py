#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_registries import TrainingReport
from reagent.evaluation.cpe import CpeEstimate


@dataclass
class DQNTrainingReport(TrainingReport):
    __registry_name__ = "dqn_report"

    td_loss: Optional[float] = None
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
class WorldModelTrainingReport(TrainingReport):
    __registry_name__ = "world_model_report"


@dataclass
class ParametricDQNTrainingReport(TrainingReport):
    __registry_name__ = "parametric_dqn_report"


@dataclass
class SlateQTrainingReport(TrainingReport):
    __registry_name__ = "slate_q_report"


@dataclass
class Seq2RewardTrainingReport(TrainingReport):
    __registry_name__ = "seq2reward_report"
