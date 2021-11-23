#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.model_managers.discrete_dqn_base import DiscreteDQNBase
from reagent.net_builder.discrete_dqn.dueling import Dueling
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import DiscreteDQNNetBuilder__Union
from reagent.reporting.discrete_dqn_reporter import DiscreteDQNReporter
from reagent.training import DQNTrainer, DQNTrainerParameters
from reagent.training import ReAgentLightningModule
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: DQNTrainerParameters = field(default_factory=DQNTrainerParameters)
    net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `Dueling`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )
    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert (
            len(self.action_names) > 1
        ), f"DiscreteDQNModel needs at least 2 actions. Got {self.action_names}."
        if self.trainer_param.minibatch_size % 8 != 0:
            logger.warn(
                f"minibatch size ({self.trainer_param.minibatch_size}) "
                "should be divisible by 8 for performance reasons!"
            )

    @property
    def action_names(self):
        return self.trainer_param.actions

    @property
    def rl_parameters(self):
        return self.trainer_param.rl

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> DQNTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            len(self.action_names),
        )

        q_network_target = q_network.get_target_network()

        reward_options = reward_options or RewardOptions()
        metrics_to_score = get_metrics_to_score(reward_options.metric_reward_values)

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.eval_parameters.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(metrics_to_score) + 1) * len(
                # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `actions`.
                self.trainer_param.actions
            )

            cpe_net_builder = self.cpe_net_builder.value
            reward_network = cpe_net_builder.build_q_network(
                self.state_feature_config,
                normalization_data_map[NormalizationKey.STATE],
                num_output_nodes,
            )
            q_network_cpe = cpe_net_builder.build_q_network(
                self.state_feature_config,
                normalization_data_map[NormalizationKey.STATE],
                num_output_nodes,
            )

            q_network_cpe_target = q_network_cpe.get_target_network()

        trainer = DQNTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=metrics_to_score,
            evaluation=self.eval_parameters,
            # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def get_reporter(self):
        return DiscreteDQNReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )

    def serving_module_names(self):
        module_names = ["default_model"]
        if len(self.action_names) == 2:
            module_names.append("binary_difference_scorer")
        return module_names

    def build_serving_modules(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ):
        assert isinstance(trainer_module, DQNTrainer)
        serving_modules = {
            "default_model": self.build_serving_module(
                trainer_module, normalization_data_map
            )
        }
        if len(self.action_names) == 2:
            serving_modules.update(
                {
                    "binary_difference_scorer": self._build_binary_difference_scorer(
                        trainer_module.q_network, normalization_data_map
                    )
                }
            )
        return serving_modules

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert isinstance(trainer_module, DQNTrainer)

        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer_module.q_network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )

    def _build_binary_difference_scorer(
        self,
        network,
        normalization_data_map: Dict[str, NormalizationData],
    ):
        assert network is not None
        net_builder = self.net_builder.value
        return net_builder.build_binary_difference_scorer(
            network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
