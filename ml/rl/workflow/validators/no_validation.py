#!/usr/bin/env python3

from dataclasses import dataclass

from ml.rl.workflow.result_types import NoValidationResults, ValidationResults
from ml.rl.workflow.types import RLTrainingOutput, ValidationOutput
from ml.rl.workflow.validators.model_validator import ModelValidator


@dataclass
class NoValidation(ModelValidator):
    """
    This is an example of how to create a validator. This validator performs no
    validation. In your own validator, you would want to have `validate()` performs
    some validation.
    """

    def validate(self, training_output: RLTrainingOutput) -> ValidationOutput:
        return ValidationOutput(
            should_publish=True,
            results=ValidationResults(no_validation_results=NoValidationResults()),
        )
