#!/usr/bin/env python3

from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.core.result_types import NoPublishingResults
from reagent.core.types import RecurringPeriod, RLTrainingOutput
from reagent.publishers.model_publisher import ModelPublisher
from reagent.workflow.model_managers.model_manager import ModelManager


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
        recurring_workflow_id: int,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> NoPublishingResults:
        return NoPublishingResults(success=True)
