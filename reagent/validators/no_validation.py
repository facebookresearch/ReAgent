#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
from typing import List, Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_types import NoValidationResults
from reagent.validators.model_validator import ModelValidator

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import RLTrainingOutput, TableSpec


@dataclass
class NoValidation(ModelValidator):
    """
    This is an example of how to create a validator. This validator performs no
    validation. In your own validator, you would want to have `validate()` performs
    some validation.
    """

    def do_validate(
        self,
        # pyre-fixme[11]: Annotation `RLTrainingOutput` is not defined as a type.
        training_output: RLTrainingOutput,
        result_history: Optional[List[RLTrainingOutput]] = None,
        # pyre-fixme[11]: Annotation `TableSpec` is not defined as a type.
        input_table_spec: Optional[TableSpec] = None,
    ) -> NoValidationResults:
        return NoValidationResults(should_publish=True)
