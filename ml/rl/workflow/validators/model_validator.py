#!/usr/bin/env python3

import abc
import logging

from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.workflow.types import RLTrainingOutput, ValidationOutput


logger = logging.getLogger(__name__)


class ModelValidator(metaclass=RegistryMeta):
    """
    Base class for model validator. All validator should subclass from this so that
    they can be registered in the workflows.
    """

    @abc.abstractmethod
    def validate(self, training_output: RLTrainingOutput) -> ValidationOutput:
        """
        This method takes RLTrainingOutput so that it can extract anything it
        might need from it.
        """
        pass
