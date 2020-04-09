#!/usr/bin/env python3

from typing import Optional

from ml.rl.core.dataclasses import dataclass
from ml.rl.evaluation.cpe import CpeEstimate
from ml.rl.workflow.result_registries import TrainingReport


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
