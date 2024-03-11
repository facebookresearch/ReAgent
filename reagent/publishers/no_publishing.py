#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import Dict, Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_types import NoPublishingResults
from reagent.model_managers.model_manager import ModelManager
from reagent.publishers.model_publisher import ModelPublisher
from reagent.workflow.types import (
    ModuleNameToEntityId,
    RecurringPeriod,
    RLTrainingOutput,
)


@dataclass
class NoPublishing(ModelPublisher):
    """
    This is an example of how to create a publisher. This publisher performs no
    publishing. In your own publisher, you would want to have `validate()` performs
    some publishing.
    """

    def do_publish(
        self,
        model_manager: ModelManager,
        training_output: RLTrainingOutput,
        setup_data: Optional[Dict[str, bytes]],
        recurring_workflow_ids: ModuleNameToEntityId,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> NoPublishingResults:
        return NoPublishingResults(success=True)
