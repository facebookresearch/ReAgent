#!/usr/bin/env python3

import abc
import dataclasses
import logging
import time
from typing import Dict, Optional, Tuple

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.tensorboardX import summary_writer_context
from reagent.training.rl_trainer_pytorch import RLTrainer
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
    7. `_set_normalization_parameters` sets the normalization parameters
    """

    def __init__(self):
        super().__init__()
        # State normalization parameters is here for convenient
        self._state_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None
        self._normalization_data_map: Optional[Dict[str, NormalizationData]] = None
        # The initialization of these attributes is delayed to `initialize_trainer()`
        self._reward_options: RewardOptions = None
        self._trainer: Optional[RLTrainer] = None
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
        match the keys expected by `_set_normalization_parameters()`
        """
        pass

    @abc.abstractmethod
    def _set_normalization_parameters(
        self, normalization_data_map: Dict[str, NormalizationData]
    ):
        """
        Set normalization parameters on current instance
        """
        pass

    @property
    def state_normalization_parameters(self) -> Dict[int, NormalizationParameters]:
        assert (
            self._state_normalization_parameters is not None
        ), "You need to set state_normalization_parameters before calling this"
        # pyre-fixme[7]: Expected `Dict[int, NormalizationParameters]` but got
        #  `Optional[Dict[int, NormalizationParameters]]`.
        # pyre-fixme[7]: Expected `Dict[int, NormalizationParameters]` but got
        #  `Optional[Dict[int, NormalizationParameters]]`.
        return self._state_normalization_parameters

    @state_normalization_parameters.setter
    def state_normalization_parameters(self, p: Dict[int, NormalizationParameters]):
        assert (
            self._state_normalization_parameters is None
        ), "You should not reset state_normalization_parameters after assignment"
        self._state_normalization_parameters = p

    def set_normalization_data_map(
        self, normalization_data_map: Dict[str, NormalizationData]
    ) -> None:
        assert (
            self._normalization_data_map is None
        ), "Cannot reset self._normalization_data_map"
        self._normalization_data_map = normalization_data_map

    def get_normalization_data(self, key: str) -> NormalizationData:
        assert (
            self._normalization_data_map is not None
        ), "self._normalization_data_map has not been set"
        assert (
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            key
            in self._normalization_data_map
        ), f"{key} not available; available keys {self._normalization_data_map.keys()}"
        return self._normalization_data_map[key]

    def get_float_features_normalization_parameters(
        self, key: str
    ) -> Dict[int, NormalizationParameters]:
        norm_data = self.get_normalization_data(key)
        dense_norm_params = norm_data.dense_normalization_parameters
        assert (
            dense_norm_params is not None
        ), f"dense_normalization_parameters for '{key}' is not set"
        return dense_norm_params

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
    def trainer(self) -> RLTrainer:
        assert self._trainer is not None, "Call initialize_trainer() first"
        # pyre-fixme[7]: Expected `RLTrainer` but got `Optional[RLTrainer]`.
        # pyre-fixme[7]: Expected `RLTrainer` but got `Optional[RLTrainer]`.
        return self._trainer

    def initialize_trainer(
        self,
        use_gpu: bool,
        reward_options: RewardOptions,
        normalization_data_map: Dict[str, NormalizationData],
        warmstart_path: Optional[str] = None,
    ) -> RLTrainer:
        """
        Initialize the trainer. Subclass should not override this. Instead,
        subclass should implement `_set_normalization_parameters()` and
        `build_trainer()`.
        """
        assert self._trainer is None, "Trainer was intialized"
        self._use_gpu = use_gpu
        self.reward_options = reward_options
        self._set_normalization_parameters(normalization_data_map)
        self._trainer = self.build_trainer()
        if warmstart_path is not None:
            trainer_state = torch.load(warmstart_path)
            # pyre-fixme[16]: `Optional` has no attribute `load_state_dict`.
            # pyre-fixme[16]: `Optional` has no attribute `load_state_dict`.
            self._trainer.load_state_dict(trainer_state)
        # pyre-fixme[7]: Expected `RLTrainer` but got `Optional[RLTrainer]`.
        # pyre-fixme[7]: Expected `RLTrainer` but got `Optional[RLTrainer]`.
        return self._trainer

    @abc.abstractmethod
    def build_trainer(self) -> RLTrainer:
        """
        Implement this to build the trainer, given the config
        """
        pass

    def train_workflow(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        normalization_data_map: Dict[str, NormalizationData],
        model,  # reagent.workflow.model_managers.ModelManager__Union
        num_epochs: int,
        use_gpu: bool,
        parent_workflow_id: int,
        child_workflow_id: int,
        reward_options: Optional[RewardOptions] = None,
        warmstart_path: Optional[str] = None,
    ) -> RLTrainingOutput:
        manager = model.value

        writer = SummaryWriter()
        logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

        warmstart_input_path = warmstart_path or None
        manager.initialize_trainer(
            use_gpu=use_gpu,
            reward_options=reward_options,
            normalization_data_map=normalization_data_map,
            warmstart_path=warmstart_input_path,
        )

        with summary_writer_context(writer):
            train_output = manager.train(train_dataset, eval_dataset, num_epochs)

        # TODO: make this a parameter
        torchscript_output_path = f"model_{round(time.time())}.torchscript"
        serving_module = manager.build_serving_module()
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
