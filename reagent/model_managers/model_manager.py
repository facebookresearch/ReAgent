#!/usr/bin/env python3

import abc
import logging
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.reporting.reporter_base import ReporterBase
from reagent.training import ReAgentLightningModule, MultiStageTrainer
from reagent.workflow.types import (
    Dataset,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from reagent.workflow.types import RLTrainingReport
from reagent.workflow.utils import get_rank, train_eval_lightning


logger = logging.getLogger(__name__)


@dataclass
class ModelManager:
    """
    ModelManager manages how to train models.

    Each type of models can have their own config type, implemented as
    `config_type()` class method. `__init__()` of the concrete class must take
    this type.

    To integrate training algorithms into the standard training workflow, you need:
    1. `build_trainer()`: Builds the ReAgentLightningModule
    2. `get_data_module()`: Defines how to create data module for this algorithm
    3. `build_serving_modules()`: Creates the TorchScript modules for serving
    4. `get_reporter()`: Returns the reporter to collect training/evaluation metrics
    5. `create_policy()`: (Optional) Creates Policy object for to interact with Gym
    """

    def __post_init_post_parse__(self):
        """
        We use pydantic to parse raw config into typed (dataclass) config.
        This method is called after everything is parsed, so you could
        validate constraints that may not be captured with the type alone.

        See https://pydantic-docs.helpmanual.io/usage/dataclasses/#initialize-hooks
        """
        pass

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ResourceOptions] = None,
    ) -> Optional[ReAgentDataModule]:
        """
        Return the data module. If this is not None, then `run_feature_identification` &
        `query_data` will not be run.
        """
        return None

    @abc.abstractmethod
    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> ReAgentLightningModule:
        """
        Implement this to build the trainer, given the config

        TODO: This function should return ReAgentLightningModule &
        the dictionary of modules created
        """
        pass

    @abc.abstractmethod
    def get_reporter(self) -> ReporterBase:
        pass

    def train(
        self,
        trainer_module: ReAgentLightningModule,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[RLTrainingOutput, pl.Trainer]:
        """
        Train the model

        Returns partially filled RLTrainingOutput.
        The field that should not be filled are:
        - output_path

        Arguments:
            train/eval/test_dataset: what you'd expect
            data_module: [pytorch lightning only] a lightning data module that replaces the use of train/eval datasets
            num_epochs: number of training epochs
            reader_options: options for the data reader
            resource_options: options for training resources (currently only used for setting num_nodes in pytorch lightning trainer)
        """
        if isinstance(trainer_module, MultiStageTrainer):
            assert trainer_module.multi_stage_total_epochs == num_epochs, (
                f"The sum of each stage's epoch ({trainer_module.trainer_epoch_mapping})"
                f" should be equal to num_epochs ({num_epochs})."
            )

        reporter = self.get_reporter()
        trainer_module.set_reporter(reporter)
        assert data_module

        lightning_trainer = train_eval_lightning(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            trainer_module=trainer_module,
            data_module=data_module,
            num_epochs=num_epochs,
            logger_name=str(type(self)),
            reader_options=reader_options,
            checkpoint_path=checkpoint_path,
            resource_options=resource_options,
        )

        rank = get_rank()
        if rank == 0:
            # pyre-ignore
            trainer_logger = lightning_trainer.logger
            logger_data = trainer_logger.line_plot_aggregated
            trainer_logger.clear_local_data()
            if reporter is None:
                training_report = None
            else:
                # pyre-ignore
                training_report = RLTrainingReport.make_union_instance(
                    reporter.generate_training_report()
                )
            return (
                RLTrainingOutput(
                    training_report=training_report, logger_data=logger_data
                ),
                lightning_trainer,
            )
        # Output from processes with non-0 rank is not used
        return RLTrainingOutput(), lightning_trainer

    # TODO: make abstract
    def build_serving_modules(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> Dict[str, torch.nn.Module]:
        """
        Returns TorchScript for serving in production
        """
        return {
            "default_model": self.build_serving_module(
                trainer_module, normalization_data_map
            )
        }

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        """
        Optionaly, implement this method if you only have one model for serving
        """
        raise NotImplementedError

    # TODO: make abstract
    def serving_module_names(self) -> List[str]:
        """
        Returns the keys that would be returned in `build_serving_modules()`.
        This method is required because we need to reserve entity IDs for
        these serving modules before we start the training.
        """
        return ["default_model"]

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ):
        raise NotImplementedError
