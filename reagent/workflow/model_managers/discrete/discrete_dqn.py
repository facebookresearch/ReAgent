#!/usr/bin/env python3

import logging
from typing import Dict

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import RewardOptions
from reagent.net_builder.discrete_dqn.dueling import Dueling
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import DiscreteDQNNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.training import DQNTrainer, DQNTrainerParameters
from reagent.training.trainer import Trainer
from reagent.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: DQNTrainerParameters = field(default_factory=DQNTrainerParameters)
    net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `Dueling`.
        # pyre-fixme[28]: Unexpected keyword argument `Dueling`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )
    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )
    # TODO: move evaluation parameters to here from trainer_param.evaluation
    # note that only DiscreteDQN and QRDQN call RLTrainer._initialize_cpe,
    # so maybe can be removed from the RLTrainer class.

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

        assert (
            len(self.trainer_param.actions) > 1
        ), f"DiscreteDQNModel needs at least 2 actions. Got {self.trainer_param.actions}."
        if self.trainer_param.minibatch_size % 8 != 0:
            logger.warn(
                f"minibatch size ({self.trainer_param.minibatch_size}) "
                "should be divisible by 8 for performance reasons!"
            )

    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> DQNTrainer:
        state_normalization_data = normalization_data_map["state"]
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_feature_config,
            state_normalization_data,
            # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `actions`.
            len(self.trainer_param.actions),
        )

        if use_gpu:
            q_network = q_network.cuda()

        q_network_target = q_network.get_target_network()

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `evaluation`.
        # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `evaluation`.
        if self.eval_parameters.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(self.metrics_to_score(reward_options)) + 1) * len(
                # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `actions`.
                self.trainer_param.actions
            )

            cpe_net_builder = self.cpe_net_builder.value
            reward_network = cpe_net_builder.build_q_network(
                self.state_feature_config, state_normalization_data, num_output_nodes
            )
            q_network_cpe = cpe_net_builder.build_q_network(
                self.state_feature_config, state_normalization_data, num_output_nodes
            )

            if use_gpu:
                reward_network.cuda()
                q_network_cpe.cuda()

            q_network_cpe_target = q_network_cpe.get_target_network()

        trainer = DQNTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score(reward_options),
            use_gpu=use_gpu,
            evaluation=self.eval_parameters,
            # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def build_serving_module(
        self, normalization_data_map: Dict[str, NormalizationData], trainer: DQNTrainer
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert trainer.q_network is not None, "_q_network was not initialized"

        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer.q_network,
            normalization_data_map["state"],
            # pyre-fixme[16]: `DQNTrainerParameters` has no attribute `actions`.
            action_names=self.trainer_param.actions,
            state_feature_config=self.state_feature_config,
        )
