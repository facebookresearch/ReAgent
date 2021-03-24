#!/usr/bin/env python3

import abc
import inspect
from typing import Dict, Optional

from reagent.core.registry_meta import RegistryMeta
from reagent.core.result_registries import PublishingResult
from reagent.model_managers.model_manager import ModelManager
from reagent.workflow.types import (
    ModuleNameToEntityId,
    RecurringPeriod,
    RLTrainingOutput,
)


class ModelPublisher(metaclass=RegistryMeta):
    """
    Base class for model publisher. All publishers should subclass from this so that
    they can be registered in the workflows.
    """

    def publish(
        self,
        model_manager: ModelManager,
        training_output: RLTrainingOutput,
        setup_data: Optional[Dict[str, bytes]],
        # Mapping from serving_module name -> recurring_workflow_id
        recurring_workflow_ids: ModuleNameToEntityId,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ):
        """
        This method takes RLTrainingOutput so that it can extract anything it
        might need from it.

        ModelManager is given here so that config can be shared
        """
        result = self.do_publish(
            model_manager,
            training_output,
            setup_data,
            recurring_workflow_ids,
            child_workflow_id,
            recurring_period,
        )
        # Avoid circular dependency at import time
        from reagent.workflow.types import PublishingResult__Union

        # We need to use inspection because the result can be a future when running on
        # FBL
        result_type = inspect.signature(self.do_publish).return_annotation
        assert result_type != inspect.Signature.empty
        # pyre-fixme[16]: `PublishingResult__Union` has no attribute
        #  `make_union_instance`.
        # pyre-fixme[16]: `PublishingResult__Union` has no attribute
        #  `make_union_instance`.
        return PublishingResult__Union.make_union_instance(result, result_type)

    @abc.abstractmethod
    def do_publish(
        self,
        model_manager: ModelManager,
        training_output: RLTrainingOutput,
        setup_data: Optional[Dict[str, bytes]],
        recurring_workflow_ids: ModuleNameToEntityId,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> PublishingResult:
        """
        This method takes RLTrainingOutput so that it can extract anything it
        might need from it.

        ModelManager is given here so that config can be shared
        """
        pass
