#!/usr/bin/env python3

import abc
import dataclasses
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.parameters import NormalizationData
from reagent.tensorboardX import summary_writer_context
from reagent.training.trainer import Trainer
from reagent.workflow.types import Dataset, RewardOptions, RLTrainingOutput, TableSpec
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
        self._trainer = self.build_trainer()
        if warmstart_path is not None:
            trainer_state = torch.load(warmstart_path)
            # pyre-fixme[16]: `Optional` has no attribute `load_state_dict`.
            # pyre-fixme[16]: `Optional` has no attribute `load_state_dict`.
            self._trainer.load_state_dict(trainer_state)
        # pyre-fixme[7]: Expected `Trainer` but got `Optional[Trainer]`.
        # pyre-fixme[7]: Expected `Trainer` but got `Optional[Trainer]`.
        return self._trainer

    @abc.abstractmethod
    def build_trainer(self) -> Trainer:
        """
        Implement this to build the trainer, given the config
        """
        pass

    def train_workflow(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        normalization_data_map: Dict[str, NormalizationData],
        num_epochs: int,
        use_gpu: bool,
        parent_workflow_id: int,
        child_workflow_id: int,
        reward_options: Optional[RewardOptions] = None,
        warmstart_path: Optional[str] = None,
    ) -> RLTrainingOutput:
        writer = SummaryWriter()
        logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

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

        with summary_writer_context(writer):
            train_output = self.train(train_dataset, eval_dataset, num_epochs)

        # TODO: make this a parameter
        torchscript_output_path = f"model_{round(time.time())}.torchscript"
        serving_module = self.build_serving_module()
        torch.jit.save(serving_module, torchscript_output_path)
        logger.info(f"Saved torchscript model to {torchscript_output_path}")
        return dataclasses.replace(train_output, output_path=torchscript_output_path)

    @abc.abstractmethod
    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
    ) -> RLTrainingOutput:
        """
        Train the model
        """
        pass

    @abc.abstractmethod
    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns TorchScript module to be used in predictor
        """
        pass

    def save_trainer(self, output_path: str) -> None:
        """
        Save the trainer for warmstarting/checkpointing.
        """
        trainer_state = self.trainer.state_dict()
        torch.save(trainer_state, output_path)
