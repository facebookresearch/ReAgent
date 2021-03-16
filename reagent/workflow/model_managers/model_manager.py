#!/usr/bin/env python3

import abc
import logging
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData
from reagent.core.registry_meta import RegistryMeta
from reagent.core.tensorboardX import summary_writer_context
from reagent.training import ReAgentLightningModule, Trainer
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.types import (
    Dataset,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelManager(metaclass=RegistryMeta):
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


    DEPRECATED: The comment below is outdated. We keep it for the context while
    migrating.

    ModelManager abstracts over common phases of training, i.e.,:
    1. `run_feature_identification()` defines how to derive feature preprocessing
       parameters from given data.
    2. `query_data()` massages the input table into the format expected by the trainer
    3. `initialize_trainer()` creates the trainer
    4. `train()`
    5. `build_serving_module()` builds the module for prediction
    6. `save_tainer()` saves the trainer for warmstarting
    """

    def __post_init_post_parse__(self):
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
        return self._use_gpu

    @property
    def reward_options(self) -> RewardOptions:
        assert self._reward_options is not None
        return self._reward_options

    @reward_options.setter
    def reward_options(self, reward_options: RewardOptions):
        assert self._reward_options is None
        # pyre-fixme[16]: `ModelManager` has no attribute `_reward_options`.
        self._reward_options = reward_options

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
    ) -> Optional[ReAgentDataModule]:
        """
        Return the data module. If this is not None, then `run_feature_identification` &
        `query_data` will not be run.
        """
        return None

    @abc.abstractmethod
    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        """
        DEPRECATED: Implement get_data_module() instead

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
        """
        DEPRECATED: Implement get_data_module() instead
        """
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        """
        DEPRECATED: Implement get_data_module() instead

        Massage input table into the format expected by the trainer
        """
        pass

    @property
    def trainer(self) -> Trainer:
        """
        DEPRECATED: The build_trainer() function should also return
        a dictionary of created networks so that other functions can
        refer to them.

        Get access to the training module. This is mostly used to extract networks
        in build_serving_modules() & create_policy().
        """
        assert self._trainer is not None, "Call initialize_trainer() first"
        return self._trainer

    def initialize_trainer(
        self,
        use_gpu: bool,
        reward_options: RewardOptions,
        normalization_data_map: Dict[str, NormalizationData],
        warmstart_path: Optional[str] = None,
    ) -> Trainer:
        """
        DEPRECATED: This should be baked into the train() function.
        `normalization_data_map` is used in build_serving_modules().
        We can pass it there directly.

        Initialize the trainer. Subclass should not override this. Instead,
        subclass should implement `required_normalization_keys()` and
        `build_trainer()`.
        """
        assert self._trainer is None, "Trainer was intialized"
        # pyre-fixme[16]: `ModelManager` has no attribute `_use_gpu`.
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
        # pyre-fixme[16]: `ModelManager` has no attribute `_normalization_data_map`.
        self._normalization_data_map = normalization_data_map
        trainer = self.build_trainer()
        # pyre-fixme[16]: `ModelManager` has no attribute `_trainer`.
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

        TODO: This function should return ReAgentLightningModule &
        the dictionary of modules created
        """
        pass

    def destroy_trainer(self):
        self._trainer = None

    @abc.abstractmethod
    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions],
    ) -> RLTrainingOutput:
        """
        DEPRECATED: Delete this once every trainer is built on PyTorch Lightning &
        every ModelManager implemnts get_data_module(). Then, we can just move the code
        in train() of DiscreteDQNBase into the training workflow function

        Train the model
        Arguments:
            train/eval/test_dataset: what you'd expect
            data_module: [pytorch lightning only] a lightning data module that replaces the use of train/eval datasets
            num_epochs: number of training epochs
            reader_options: options for the data reader
            resource_options: options for training resources (currently only used for setting num_nodes in pytorch lightning trainer)
        """
        pass

    # TODO: make abstract
    # TODO: This function should take normalization_data_map &
    # dictionary of modules created in `build_trainer()`
    def build_serving_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Returns TorchScript for serving in production
        """
        return {"default_model": self.build_serving_module()}

    # TODO: make abstract
    def serving_module_names(self) -> List[str]:
        """
        Returns the keys that would be returned in `build_serving_modules()`.
        This method is required because we need to reserve entity IDs for
        these serving modules before we start the training.
        """
        return ["default_model"]
