#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import Dict, Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_types import NoPublishingResults
from reagent.model_managers.model_manager import ModelManager
from reagent.publishers.model_publisher import ModelPublisher

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
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
        # pyre-fixme[11]: Annotation `RLTrainingOutput` is not defined as a type.
        training_output: RLTrainingOutput,
        setup_data: Optional[Dict[str, bytes]],
        # pyre-fixme[11]: Annotation `ModuleNameToEntityId` is not defined as a type.
        recurring_workflow_ids: ModuleNameToEntityId,
        child_workflow_id: int,
        # pyre-fixme[11]: Annotation `RecurringPeriod` is not defined as a type.
        recurring_period: Optional[RecurringPeriod],
    ) -> NoPublishingResults:
        return NoPublishingResults(success=True)
