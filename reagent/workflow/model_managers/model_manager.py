#!/usr/bin/env python3

import abc
import dataclasses
import logging
import time
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from fvcore.common.file_io import PathManager
from reagent.core.registry_meta import RegistryMeta
from reagent.parameters import NormalizationData
from reagent.tensorboardX import summary_writer_context
from reagent.training import ReAgentLightningModule, Trainer
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.types import (
    Dataset,
    ModuleNameToEntityId,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class ModelManager(metaclass=RegistryMeta):
    """
    ModelManager manages how to train models.

    Each type of models can have their own config type, implemented as
    `config_type()` class method. `__init__()` of the concrete class must take
    this type.

    ModelManager abstracts over common phases of training, i.e.,:
    1. `run_feature_identification()` defines how to derive feature preprocessing
       parameters from given data.
    2. `query_data()` massages the input table into the format expected by the trainer
    3. `initialize_trainer()` creates the trainer
    4. `train()`
    5. `build_serving_module()` builds the module for prediction
    6. `save_tainer()` saves the trainer for warmstarting
    """

    def __init__(self):
        super().__init__()
        # initialization is delayed to `initialize_trainer()`
        self._normalization_data_map: Optional[Dict[str, NormalizationData]] = None
        self._reward_options: Optional[RewardOptions] = None
        self._trainer: Optional[Trainer] = None
        self._use_gpu: Optional[bool] = None
        self._lightning_trainer: Optional[pl.Trainer] = None
        self._lightning_checkpoint_path: Optional[str] = None

    @property
    def use_gpu(self) -> bool:
        assert (
            self._use_gpu is not None
        ), "Call initialize_trainer() to set the value first"
        # pyre-fixme[7]: Expected `bool` but got `Optional[bool]`.
        # pyre-fixme[7]: Expected `bool` but got `Optional[bool]`.
        return self._use_gpu

    @property
    def reward_options(self) -> RewardOptions:
        assert self._reward_options is not None
        # pyre-fixme[7]: Expected `RewardOptions` but got `Optional[RewardOptions]`.
        # pyre-fixme[7]: Expected `RewardOptions` but got `Optional[RewardOptions]`.
        return self._reward_options

    @reward_options.setter
    def reward_options(self, reward_options: RewardOptions):
        assert self._reward_options is None
        self._reward_options = reward_options

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
    ) -> Optional[ReAgentDataModule]:
        # Return the data module. If this is not None, then `run_feature_identification` &
        # `query_data` will not be run.
        return None

    @abc.abstractmethod
    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        """
        Derive preprocessing parameters from data. The keys of the dict should
        match the keys from `required_normalization_keys()`
        """
        pass

    @property
    @abc.abstractmethod
    def required_normalization_keys(self) -> List[str]:
        """ Get the normalization keys required for current instance """
        pass

    def __getattr__(self, attr):
        """ Get X_normalization_data by attribute """
        normalization_data_suffix = "_normalization_data"
        if attr.endswith(normalization_data_suffix):
            assert self._normalization_data_map is not None, (
                f"Trying to access {attr} but normalization_data_map "
                "has not been set via `initialize_trainer`."
            )
            normalization_key = attr[: -len(normalization_data_suffix)]
            normalization_data = self._normalization_data_map.get(
                normalization_key, None
            )
            if normalization_data is None:
                raise AttributeError(
                    f"normalization key `{normalization_key}` is unavailable. "
                    f"Available keys are: {self._normalization_data_map.keys()}."
                )
            return normalization_data

        raise AttributeError(
            f"attr {attr} not available {type(self)} (subclass of ModelManager)."
        )

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        """
        Massage input table into the format expected by the trainer
        """
        pass

    @property
    def trainer(self) -> Trainer:
        assert self._trainer is not None, "Call initialize_trainer() first"
        # pyre-fixme[7]: Expected `Trainer` but got `Optional[Trainer]`.
        # pyre-fixme[7]: Expected `Trainer` but got `Optional[Trainer]`.
        return self._trainer

    def initialize_trainer(
        self,
        use_gpu: bool,
        reward_options: RewardOptions,
        normalization_data_map: Dict[str, NormalizationData],
        warmstart_path: Optional[str] = None,
    ) -> Trainer:
        """
        Initialize the trainer. Subclass should not override this. Instead,
        subclass should implement `required_normalization_keys()` and
        `build_trainer()`.
        """
        assert self._trainer is None, "Trainer was intialized"
        self._use_gpu = use_gpu
        self.reward_options = reward_options
        # validate that we have all the required keys
        for normalization_key in self.required_normalization_keys:
            normalization_data = normalization_data_map.get(normalization_key, None)
            assert normalization_data is not None, (
                f"NormalizationData for {normalization_key} "
                "is required but not provided."
            )
            # NOTE: Don't need this check in the future, for non-dense parameters
            assert normalization_data.dense_normalization_parameters is not None, (
                f"Dense normalization parameters for "
                f"{normalization_key} is not provided."
            )
        assert (
            self._normalization_data_map is None
        ), "Cannot reset self._normalization_data_map"
        self._normalization_data_map = normalization_data_map
        trainer = self.build_trainer()
        self._trainer = trainer
        if warmstart_path is not None:
            # pyre-fixme[16]: Module `pl` has no attribute `LightningModule`.
            # pyre-fixme[16]: Module `pl` has no attribute `LightningModule`.
            if isinstance(trainer, pl.LightningModule):
                # Delayed until Trainer is initialized
                self._lightning_checkpoint_path = warmstart_path
            else:
                trainer_state = torch.load(warmstart_path)
                trainer.load_state_dict(trainer_state)
        return trainer

    @abc.abstractmethod
    def build_trainer(self) -> Trainer:
        """
        Implement this to build the trainer, given the config
        """
        pass

    def train_workflow(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        *,
        num_epochs: int,
        use_gpu: bool,
        named_model_ids: ModuleNameToEntityId,
        child_workflow_id: int,
        setup_data: Optional[Dict[str, bytes]] = None,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
        reward_options: Optional[RewardOptions] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ResourceOptions] = None,
        warmstart_path: Optional[str] = None,
    ) -> RLTrainingOutput:
        writer = SummaryWriter()
        logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

        if setup_data is not None:
            data_module = self.get_data_module(
                setup_data=setup_data, reader_options=reader_options
            )
            assert data_module is not None
            data_module.setup()
        else:
            data_module = None

        if normalization_data_map is None:
            assert data_module is not None
            normalization_data_map = data_module.get_normalization_data_map(
                self.required_normalization_keys
            )

        warmstart_input_path = warmstart_path or None
        self.initialize_trainer(
            use_gpu=use_gpu,
            # pyre-fixme[6]: Expected `RewardOptions` for 2nd param but got
            #  `Optional[RewardOptions]`.
            # pyre-fixme[6]: Expected `RewardOptions` for 2nd param but got
            #  `Optional[RewardOptions]`.
            reward_options=reward_options,
            normalization_data_map=normalization_data_map,
            warmstart_path=warmstart_input_path,
        )

        if not reader_options:
            reader_options = ReaderOptions()

        with summary_writer_context(writer):
            train_output = self.train(
                train_dataset, eval_dataset, data_module, num_epochs, reader_options
            )

        output_paths = {}
        for module_name, serving_module in self.build_serving_modules().items():
            # TODO: make this a parameter
            torchscript_output_path = f"model_{round(time.time())}.torchscript"
            serving_module = self.build_serving_module()
            torch.jit.save(serving_module, torchscript_output_path)
            logger.info(f"Saved {module_name} to {torchscript_output_path}")
            output_paths[module_name] = torchscript_output_path
        return dataclasses.replace(train_output, output_paths=output_paths)

    @abc.abstractmethod
    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
    ) -> RLTrainingOutput:
        """
        Train the model
        """
        pass

    # TODO: make abstract
    def build_serving_modules(self) -> Dict[str, torch.nn.Module]:
        # eventually move to this method to be more generic
        return {"default_model": self.build_serving_module()}

    # TODO: make abstract
    def serving_module_names(self) -> List[str]:
        # should match sorted(self.build_serving_modules.keys())
        return ["default_model"]

    def save_trainer(self, output_path: str) -> None:
        """
        Save the trainer for warmstarting/checkpointing.
        """
        lightning_trainer = self._lightning_trainer
        if lightning_trainer:
            trainer = self.trainer
            assert isinstance(trainer, ReAgentLightningModule)
            trainer._cleanly_stopped[0] = True
            # HACK: since lightning_trainer.save_checkpoint can only deal with
            # local file paths (not even file handlers), we save to local file
            # first, and then use PathManager
            local_path = "/tmp/lightning_save_checkpoint_local_copy"
            lightning_trainer.save_checkpoint(local_path)
            with open(local_path, "rb") as local_f:
                checkpoint_contents = local_f.read()
            with PathManager.open(output_path, "wb") as output_f:
                output_f.write(checkpoint_contents)
        else:
            trainer_state = self.trainer.state_dict()
            torch.save(trainer_state, output_path)
