#!/usr/bin/env python3

import abc
import logging
from typing import Dict, List, Optional, Tuple

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.core.types import Dataset, ReaderOptions, RewardOptions, TableSpec
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.parameters import NormalizationData
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.training.trainer import Trainer


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
    6. `save_trainer()` saves the trainer for warmstarting
    """

    @abc.abstractmethod
    def run_feature_identification(
        self, data_fetcher: DataFetcher, input_table_spec: TableSpec
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

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        raise NotImplementedError()

    def get_evaluator(self, trainer, reward_options: RewardOptions):
        return None

    @abc.abstractmethod
    def query_data(
        self,
        data_fetcher: DataFetcher,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        """
        Massage input table into the format expected by the trainer
        """
        pass

    @abc.abstractmethod
    def get_reporter(self):
        """
        Get the reporter that displays statistics after training
        """
        pass

    @abc.abstractmethod
    def build_batch_preprocessor(
        self,
        reader_options: ReaderOptions,
        use_gpu: bool,
        batch_size: int,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> BatchPreprocessor:
        """
        The Batch Preprocessor is a module that transforms data to a form that can be (1) read by the trainer
        or (2) used in part of the serving module.  For training, the batch preprocessor is typically run
        on reader machines in parallel so the GPUs on the trainer machines can be fully utilized.
        """
        pass

    @abc.abstractmethod
    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> Trainer:
        """
        Implement this to build the trainer, given the config
        """
        pass

    def create_policy(self, trainer) -> Policy:
        """ Create a Policy from env. """
        raise NotImplementedError()

    def create_serving_policy(
        self, normalization_data_map: Dict[str, NormalizationData], trainer
    ) -> Policy:
        """ Create an online Policy from env. """
        return create_predictor_policy_from_model(
            self.build_serving_module(normalization_data_map, trainer)
        )

    @abc.abstractmethod
    def build_serving_module(
        self, normalization_data_map: Dict[str, NormalizationData], trainer
    ) -> torch.nn.Module:
        """
        Returns TorchScript module to be used in predictor
        """
        pass
