#!/usr/bin/env python3

from typing import Optional

from ml.rl.core.dataclasses import dataclass
from ml.rl.workflow.model_managers.model_manager import ModelManager
from ml.rl.workflow.publishers.model_publisher import ModelPublisher
from ml.rl.workflow.result_types import NoPublishingResults
from ml.rl.workflow.types import RecurringPeriod, RLTrainingOutput


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
