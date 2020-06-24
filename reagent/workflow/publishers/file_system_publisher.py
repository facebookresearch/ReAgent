#!/usr/bin/env python3

import logging
import os
from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.publishers.model_publisher import ModelPublisher
from reagent.workflow.result_types import NoPublishingResults
from reagent.workflow.types import RecurringPeriod, RLTrainingOutput


try:
    # pyre-fixme[21]: Could not find `tinydb`.
    # pyre-fixme[21]: Could not find `tinydb`.
    from tinydb import Query, TinyDB

    HAS_TINYDB = True

except ImportError:
    HAS_TINYDB = False

    class FileSystemPublisher:
        pass


logger = logging.getLogger(__name__)

KEY_FIELD = "model_config"
VALUE_FIELD = "torchscript_path"

if HAS_TINYDB:

    @dataclass
    class FileSystemPublisher(ModelPublisher):
        """ Uses a file to serve as a key-value store.
        The key is the str/repr representation of the ModelManager.
        The value is the path to the torchscipt model.

        TODO: replace with redis (python) and hiredis (C) for better RASP support
        """

        publishing_file: str = "/tmp/file_system_publisher"

        def __post_init_post_parse__(self):
            self.publishing_file = os.path.abspath(self.publishing_file)
            self.db: TinyDB = TinyDB(self.publishing_file)
            logger.info(f"Using TinyDB at {self.publishing_file}.")

        def get_latest_published_model(self, model_manager: ModelManager) -> str:
            Model = Query()
            key = str(model_manager)
            # pyre-fixme[16]: `FileSystemPublisher` has no attribute `db`.
            results = self.db.search(Model[KEY_FIELD] == key)
            if len(results) != 1:
                if len(results) == 0:
                    raise ValueError(
                        "Publish a model with the same str representation first!"
                    )
                else:
                    raise RuntimeError(
                        f"Got {len(results)} results for model_manager. {results}"
                    )
            return results[0][VALUE_FIELD]

        def do_publish(
            self,
            model_manager: ModelManager,
            training_output: RLTrainingOutput,
            recurring_workflow_id: int,
            child_workflow_id: int,
            recurring_period: Optional[RecurringPeriod],
        ) -> NoPublishingResults:
            path = training_output.output_path
            assert path is not None, f"Given path is None."
            assert os.path.exists(path), f"Given path {path} doesn't exist."
            Model = Query()
            # find if there's already been something stored
            key = str(model_manager)
            # pyre-fixme[16]: `FileSystemPublisher` has no attribute `db`.
            results = self.db.search(Model[KEY_FIELD] == key)
            if len(results) == 0:
                # this is a first
                self.db.insert({KEY_FIELD: key, VALUE_FIELD: path})
            else:
                # replace it
                if len(results) > 1:
                    raise RuntimeError(
                        f"Got {len(results)} results for model_manager. {results}"
                    )
                self.db.update({VALUE_FIELD: path}, Model[KEY_FIELD] == key)
            return NoPublishingResults(success=True)
