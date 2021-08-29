#!/usr/bin/env python3
from typing import List, Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_types import NoValidationResults
from reagent.validators.model_validator import ModelValidator
from reagent.workflow.types import RLTrainingOutput
from reagent.workflow.types import TableSpec


@dataclass
class NoValidation(ModelValidator):
    """
    This is an example of how to create a validator. This validator performs no
    validation. In your own validator, you would want to have `validate()` performs
    some validation.
    """

    def do_validate(
        self,
        training_output: RLTrainingOutput,
        result_history: Optional[List[RLTrainingOutput]] = None,
        input_table_spec: Optional[TableSpec] = None,
    ) -> NoValidationResults:
        return NoValidationResults(should_publish=True)
