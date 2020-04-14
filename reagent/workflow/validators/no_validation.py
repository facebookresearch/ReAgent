#!/usr/bin/env python3

from ml.rl.core.dataclasses import dataclass
from ml.rl.workflow.result_types import NoValidationResults
from ml.rl.workflow.types import RLTrainingOutput
from ml.rl.workflow.validators.model_validator import ModelValidator


@dataclass
class NoValidation(ModelValidator):
    """
    This is an example of how to create a validator. This validator performs no
    validation. In your own validator, you would want to have `validate()` performs
    some validation.
    """

    def do_validate(self, training_output: RLTrainingOutput) -> NoValidationResults:
        return NoValidationResults(should_publish=True)
