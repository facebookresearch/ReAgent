#!/usr/bin/env python3

import abc
from typing import Optional

from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.workflow.model_managers.model_manager import ModelManager
from ml.rl.workflow.types import PublishingOutput, RecurringPeriod, RLTrainingOutput


class ModelPublisher(metaclass=RegistryMeta):
    """
    Base class for model publisher. All publishers should subclass from this so that
    they can be registered in the workflows.
    """

    @abc.abstractmethod
    def publish(
        self,
        model_manager: ModelManager,
        training_output: RLTrainingOutput,
        recurring_workflow_id: int,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> PublishingOutput:
        """
        This method takes RLTrainingOutput so that it can extract anything it
        might need from it.

        ModelManager is given here so that config can be shared
        """
        pass
