#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.critic import FullyConnectedCritic
from reagent.models.dqn import FullyConnectedDQN
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


class DuelingQNetwork(ModelBase):
    def __init__(
        self,
        *,
        shared_network: ModelBase,
        advantage_network: ModelBase,
        value_network: ModelBase,
    ) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581
        """
        super().__init__()
        self.shared_network = shared_network
        input_prototype = shared_network.input_prototype()
        assert isinstance(
            input_prototype, rlt.FeatureData
        ), f"shared_network should expect FeatureData as input"
        self.advantage_network = advantage_network
        self.value_network = value_network

        _check_connection(self)
        self._name = "unnamed"

    @classmethod
    def make_fully_connected(
        cls,
        state_dim: int,
        action_dim: int,
        layers: List[int],
        activations: List[str],
        num_atoms: Optional[int] = None,
        use_batch_norm: bool = False,
    ):
        assert len(layers) > 0, "Must have at least one layer"
        state_embedding_dim = layers[-1]
        assert state_embedding_dim % 2 == 0, "The last size must be divisible by 2"
        shared_network = FullyConnectedDQN(
            state_dim,
            state_embedding_dim,
            sizes=layers[:-1],
            activations=activations[:-1],
            normalized_output=True,
            use_batch_norm=use_batch_norm,
        )
        advantage_network = FullyConnectedDQN(
            state_embedding_dim,
            action_dim,
            sizes=[state_embedding_dim // 2],
            activations=activations[-1:],
            num_atoms=num_atoms,
        )
        value_network = FullyConnectedDQN(
            state_embedding_dim,
            1,
            sizes=[state_embedding_dim // 2],
            activations=activations[-1:],
            num_atoms=num_atoms,
        )

        return cls(
            shared_network=shared_network,
            advantage_network=advantage_network,
            value_network=value_network,
        )

    def input_prototype(self):
        return self.shared_network.input_prototype()

    def _get_values(
        self, state: rlt.FeatureData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_state = rlt.FeatureData(self.shared_network(state))
        value = self.value_network(shared_state)
        raw_advantage = self.advantage_network(shared_state)
        reduce_over = tuple(range(1, raw_advantage.dim()))
        advantage = raw_advantage - raw_advantage.mean(dim=reduce_over, keepdim=True)

        q_value = value + advantage
        return value, raw_advantage, advantage, q_value

    def forward(self, state: rlt.FeatureData) -> torch.Tensor:
        value, raw_advantage, advantage, q_value = self._get_values(state)

        # TODO: export these as observable values
        if SummaryWriterContext._global_step % 1000 == 0:
            _log_histogram_and_mean(self._name, "value", value)
            _log_histogram_and_mean(self._name, "q_value", q_value)
            _log_histogram_and_mean(self._name, "raw_advantage", raw_advantage)
            advantage = advantage.detach()
            for i in range(advantage.shape[1]):
                a = advantage[:, i]
                _log_histogram_and_mean(f"{self._name}/{i}", "advantage", a)

        return q_value


class ParametricDuelingQNetwork(ModelBase):
    def __init__(
        self,
        *,
        shared_network: ModelBase,
        advantage_network: ModelBase,
        value_network: ModelBase,
    ) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581
        """
        super().__init__()
        advantage_network_input = advantage_network.input_prototype()
        assert (
            isinstance(advantage_network_input, tuple)
            and len(advantage_network_input) == 2
        )
        assert advantage_network_input[0].has_float_features_only

        self.shared_network = shared_network
        self.advantage_network = advantage_network
        self.value_network = value_network

        _check_connection(self)
        self._name = "unnamed"

    @classmethod
    def make_fully_connected(
        cls,
        state_dim: int,
        action_dim: int,
        layers: List[int],
        activations: List[str],
        use_batch_norm: bool = False,
    ):
        state_embedding_dim = layers[-1]
        shared_network = FullyConnectedDQN(
            state_dim,
            state_embedding_dim,
            sizes=layers[:-1],
            activations=activations[:-1],
            normalized_output=True,
        )
        advantage_network = FullyConnectedCritic(
            state_embedding_dim,
            action_dim,
            sizes=[state_embedding_dim // 2],
            activations=activations[-1:],
        )
        value_network = FullyConnectedDQN(
            state_embedding_dim,
            1,
            sizes=[state_embedding_dim // 2],
            activations=activations[-1:],
        )
        return ParametricDuelingQNetwork(
            shared_network=shared_network,
            advantage_network=advantage_network,
            value_network=value_network,
        )

    def input_prototype(self):
        shared_network_input = self.shared_network.input_prototype()
        assert isinstance(shared_network_input, rlt.FeatureData)
        _state, action = self.advantage_network.input_prototype()
        return (shared_network_input, action)

    def _get_values(
        self, state_action: Tuple[rlt.FeatureData, rlt.FeatureData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action = state_action
        shared_state = rlt.FeatureData(self.shared_network(state))

        value = self.value_network(shared_state)
        advantage = self.advantage_network(shared_state, action)
        q_value = value + advantage
        return advantage, value, q_value

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData) -> torch.Tensor:
        advantage, value, q_value = self._get_values((state, action))

        # TODO: export these as observable values
        if SummaryWriterContext._global_step % 1000 == 0:
            _log_histogram_and_mean(self._name, "value", value)
            _log_histogram_and_mean(self._name, "q_value", q_value)
            _log_histogram_and_mean(self._name, "advantage", advantage)

        return q_value


def _log_histogram_and_mean(name, key, x):
    SummaryWriterContext.add_histogram(
        f"dueling_network/{name}/{key}", x.detach().cpu()
    )
    SummaryWriterContext.add_scalar(
        f"dueling_network/{name}/mean_{key}", x.detach().mean().cpu()
    )


def _check_connection(model):
    try:
        with torch.no_grad():
            model.eval()
            _ = model._get_values(model.input_prototype())
    except Exception:
        logger.error(
            "The networks aren't connecting to each other; check your networks"
        )
        raise
    finally:
        model.train()
