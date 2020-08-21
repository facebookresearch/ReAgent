#!/usr/bin/env python3

import logging
from typing import Optional

import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.gym.policies.policy import Policy
from reagent.models.cem_planner import CEMPlannerNetwork
from reagent.parameters import CEMTrainerParameters, param_hash
from reagent.preprocessing.identify_types import CONTINUOUS_ACTION
from reagent.preprocessing.normalization import get_num_output_features
from reagent.training.cem_trainer import CEMTrainer
from reagent.workflow.model_managers.model_based.world_model import WorldModel
from reagent.workflow.model_managers.world_model_base import WorldModelBase


logger = logging.getLogger(__name__)


class CEMPolicy(Policy):
    def __init__(self, cem_planner_network: CEMPlannerNetwork, discrete_action: bool):
        self.cem_planner_network = cem_planner_network
        self.discrete_action = discrete_action

    # TODO: consider possible_actions_mask
    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[np.ndarray] = None
    ) -> rlt.ActorOutput:
        greedy = self.cem_planner_network(obs)
        if self.discrete_action:
            _, onehot = greedy
            return rlt.ActorOutput(
                action=onehot.unsqueeze(0), log_prob=torch.tensor(0.0)
            )
        else:
            return rlt.ActorOutput(
                action=greedy.unsqueeze(0), log_prob=torch.tensor(0.0)
            )


@dataclass
class CrossEntropyMethod(WorldModelBase):
    __hash__ = param_hash

    trainer_param: CEMTrainerParameters = field(default_factory=CEMTrainerParameters)

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    # TODO: should this be in base class?
    def create_policy(self, serving: bool = False) -> Policy:
        return CEMPolicy(self.cem_planner_network, self.discrete_action)

    def build_trainer(self) -> CEMTrainer:
        world_model_manager: WorldModel = WorldModel(
            trainer_param=self.trainer_param.mdnrnn
        )
        world_model_manager.initialize_trainer(
            self.use_gpu,
            self.reward_options,
            # pyre-fixme[6]: Expected `Dict[str,
            #  reagent.parameters.NormalizationData]` for 3rd param but got
            #  `Optional[typing.Dict[str, reagent.parameters.NormalizationData]]`.
            # pyre-fixme[6]: Expected `Dict[str,
            #  reagent.parameters.NormalizationData]` for 3rd param but got
            #  `Optional[typing.Dict[str, reagent.parameters.NormalizationData]]`.
            self._normalization_data_map,
        )
        world_model_trainers = [
            world_model_manager.build_trainer()
            for _ in range(self.trainer_param.num_world_models)
        ]
        world_model_nets = [trainer.memory_network for trainer in world_model_trainers]
        terminal_effective = self.trainer_param.mdnrnn.not_terminal_loss_weight > 0

        action_normalization_parameters = (
            self.action_normalization_data.dense_normalization_parameters
        )
        sorted_action_norm_vals = list(action_normalization_parameters.values())
        discrete_action = sorted_action_norm_vals[0].feature_type != CONTINUOUS_ACTION
        action_upper_bounds, action_lower_bounds = None, None
        if not discrete_action:
            action_upper_bounds = np.array(
                [v.max_value for v in sorted_action_norm_vals]
            )
            action_lower_bounds = np.array(
                [v.min_value for v in sorted_action_norm_vals]
            )

        cem_planner_network = CEMPlannerNetwork(
            mem_net_list=world_model_nets,
            cem_num_iterations=self.trainer_param.cem_num_iterations,
            cem_population_size=self.trainer_param.cem_population_size,
            ensemble_population_size=self.trainer_param.ensemble_population_size,
            num_elites=self.trainer_param.num_elites,
            plan_horizon_length=self.trainer_param.plan_horizon_length,
            state_dim=get_num_output_features(
                self.state_normalization_data.dense_normalization_parameters
            ),
            action_dim=get_num_output_features(
                self.action_normalization_data.dense_normalization_parameters
            ),
            discrete_action=discrete_action,
            terminal_effective=terminal_effective,
            gamma=self.trainer_param.rl.gamma,
            alpha=self.trainer_param.alpha,
            epsilon=self.trainer_param.epsilon,
            action_upper_bounds=action_upper_bounds,
            action_lower_bounds=action_lower_bounds,
        )
        # store for building policy
        # pyre-fixme[16]: `CrossEntropyMethod` has no attribute `discrete_action`.
        self.discrete_action = discrete_action
        # pyre-fixme[16]: `CrossEntropyMethod` has no attribute `cem_planner_network`.
        self.cem_planner_network = cem_planner_network
        logger.info(
            f"Built CEM network with discrete action = {discrete_action}, "
            f"action_upper_bound={action_upper_bounds}, "
            f"action_lower_bounds={action_lower_bounds}"
        )
        return CEMTrainer(
            cem_planner_network=cem_planner_network,
            world_model_trainers=world_model_trainers,
            parameters=self.trainer_param,
            use_gpu=self.use_gpu,
        )

    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        raise NotImplementedError()
