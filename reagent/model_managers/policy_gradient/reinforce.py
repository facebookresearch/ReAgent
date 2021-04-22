#!/usr/bin/env python3

import logging
from typing import Dict, Optional, Tuple, List

import torch
from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData
from reagent.core.parameters import NormalizationKey
from reagent.core.parameters import param_hash
from reagent.data.data_fetcher import DataFetcher
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.model_managers.model_manager import ModelManager
from reagent.models.model_feature_config_provider import RawModelFeatureConfigProvider
from reagent.net_builder.discrete_dqn.dueling import Dueling
from reagent.net_builder.unions import (
    DiscreteDQNNetBuilder__Union,
    ValueNetBuilder__Union,
)
from reagent.training import ReinforceTrainer, ReinforceTrainerParameters
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.types import (
    Dataset,
    ModelFeatureConfigProvider__Union,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)


logger = logging.getLogger(__name__)


@dataclass
class Reinforce(ModelManager):
    __hash__ = param_hash

    trainer_param: ReinforceTrainerParameters = field(
        default_factory=ReinforceTrainerParameters
    )
    # using DQN net here because it supports `possible_actions_mask`
    policy_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-ignore
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )
    value_net_builder: Optional[ValueNetBuilder__Union] = None
    state_feature_config_provider: ModelFeatureConfigProvider__Union = field(
        # pyre-ignore
        default_factory=lambda: ModelFeatureConfigProvider__Union(
            raw=RawModelFeatureConfigProvider(float_feature_infos=[])
        )
    )
    sampler_temperature: float = 1.0

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.action_names = self.trainer_param.actions
        self._policy: Optional[Policy] = None
        assert (
            len(self.action_names) > 1
        ), f"REINFORCE needs at least 2 actions. Got {self.action_names}."

    # pyre-ignore
    def build_trainer(self, use_gpu: bool) -> ReinforceTrainer:
        policy_net_builder = self.policy_net_builder.value
        # pyre-ignore
        self._policy_network = policy_net_builder.build_q_network(
            self.state_feature_config,
            self.state_normalization_data,
            len(self.action_names),
        )
        value_net = None
        if self.value_net_builder:
            value_net_builder = self.value_net_builder.value  # pyre-ignore
            value_net = value_net_builder.build_value_network(
                self.state_normalization_data
            )
        trainer = ReinforceTrainer(
            policy=self.create_policy(),
            value_net=value_net,
            **self.trainer_param.asdict(),  # pyre-ignore
        )
        return trainer

    def create_policy(self, serving: bool = False):
        if serving:
            return create_predictor_policy_from_model(self.build_serving_module())
        else:
            if self._policy is None:
                sampler = SoftmaxActionSampler(temperature=self.sampler_temperature)
                # pyre-ignore
                self._policy = Policy(scorer=self._policy_network, sampler=sampler)
            return self._policy

    def build_serving_module(self) -> torch.nn.Module:
        assert self._policy_network is not None
        policy_serving_module = self.policy_net_builder.value.build_serving_module(
            q_network=self._policy_network,
            state_normalization_data=self.state_normalization_data,
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
        return policy_serving_module

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        raise NotImplementedError

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE]

    @property
    def should_generate_eval_dataset(self) -> bool:
        raise NotImplementedError

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise NotImplementedError

    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
    ) -> RLTrainingOutput:
        raise NotImplementedError

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return self.state_feature_config_provider.value.get_model_feature_config()
