#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

from ml.rl.workflow.model_managers.model_manager import ModelManager
from ml.rl.workflow.publishers.model_publisher import ModelPublisher
from ml.rl.workflow.result_types import NoPublishingResults, PublishingResults
from ml.rl.workflow.types import PublishingOutput, RecurringPeriod, RLTrainingOutput


@dataclass
class NoPublishing(ModelPublisher):
    """
    This is an example of how to create a publisher. This publisher performs no
    publishing. In your own publisher, you would want to have `validate()` performs
    some publishing.
    """

    def publish(
        self,
        model_manager: ModelManager,
        training_output: RLTrainingOutput,
        recurring_workflow_id: int,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> PublishingOutput:
        return PublishingOutput(
            success=True,
            results=PublishingResults(no_publishing_results=NoPublishingResults()),
        )
