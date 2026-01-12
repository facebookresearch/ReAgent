#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Note: this file is modeled after td3.py

# pyre-unsafe

import logging
from typing import Dict, Optional

import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
    param_hash,
)
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.model_managers.discrete_dqn_base import DiscreteDQNBase
from reagent.models.base import ModelBase
from reagent.net_builder.discrete_actor.fully_connected import (
    FullyConnected as DiscreteFullyConnected,
)
from reagent.net_builder.discrete_dqn.dueling import Dueling
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import (
    DiscreteActorNetBuilder__Union,
    DiscreteDQNNetBuilder__Union,
)
from reagent.prediction.cfeval.predictor_wrapper import BanditRewardNetPredictorWrapper
from reagent.reporting.discrete_crr_reporter import DiscreteCRRReporter
from reagent.training import (
    CRRTrainerParameters,
    DiscreteCRRTrainer,
    ReAgentLightningModule,
)

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import RewardOptions

logger = logging.getLogger(__name__)


class ActorPolicyWrapper(Policy):
    """Actor's forward function is our act"""

    def __init__(self, actor_network):
        self.actor_network = actor_network

    @torch.no_grad()
    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        self.actor_network.eval()
        output = self.actor_network(obs)
        self.actor_network.train()
        return output.detach().cpu()


@dataclass
class DiscreteCRR(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: CRRTrainerParameters = field(default_factory=CRRTrainerParameters)

    actor_net_builder: DiscreteActorNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteActorNetBuilder__Union(
            FullyConnected=DiscreteFullyConnected()
        )
    )

    critic_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )

    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert len(self.action_names) > 1, (
            f"DiscreteCRRModel needs at least 2 actions. Got {self.action_names}."
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
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
    ) -> DiscreteCRRTrainer:
        actor_net_builder = self.actor_net_builder.value
        actor_network = actor_net_builder.build_actor(
            normalization_data_map[NormalizationKey.STATE], len(self.action_names)
        )
        actor_network_target = actor_network.get_target_network()

        # The arguments to q_network1 and q_network2 below are modeled after those in discrete_dqn.py
        critic_net_builder = self.critic_net_builder.value

        q1_network = critic_net_builder.build_q_network(
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            len(self.action_names),
        )
        q1_network_target = q1_network.get_target_network()

        q2_network = q2_network_target = None
        # pyre-fixme[16]: `CRRTrainerParameters` has no attribute
        #  `double_q_learning`.
        if self.trainer_param.double_q_learning:
            q2_network = critic_net_builder.build_q_network(
                self.state_feature_config,
                normalization_data_map[NormalizationKey.STATE],
                len(self.action_names),
            )
            q2_network_target = q2_network.get_target_network()

        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        reward_options = reward_options or RewardOptions()
        metrics_to_score = get_metrics_to_score(reward_options.metric_reward_values)

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.eval_parameters.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(metrics_to_score) + 1) * len(
                # pyre-fixme[16]: `CRRTrainerParameters` has no attribute `actions`.
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

        trainer = DiscreteCRRTrainer(
            actor_network=actor_network,
            actor_network_target=actor_network_target,
            q1_network=q1_network,
            q1_network_target=q1_network_target,
            reward_network=reward_network,
            q2_network=q2_network,
            q2_network_target=q2_network_target,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=metrics_to_score,
            evaluation=self.eval_parameters,
            # pyre-fixme[16]: `CRRTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ) -> Policy:
        """Create online actor critic policy."""
        assert isinstance(trainer_module, DiscreteCRRTrainer)
        if serving:
            assert normalization_data_map
            return create_predictor_policy_from_model(
                self.build_actor_module(trainer_module, normalization_data_map)
            )
        else:
            return ActorPolicyWrapper(trainer_module.actor_network)

    def get_reporter(self):
        return DiscreteCRRReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )

    # Note: when using test_gym.py as the entry point, the normalization data
    # is set when the line     normalization = build_normalizer(env)   is executed.
    # The code then calls build_state_normalizer() and build_action_normalizer()
    # in utils.py

    def serving_module_names(self):
        module_names = ["default_model", "dqn", "actor_dqn"]
        if len(self.action_names) == 2:
            module_names.append("binary_difference_scorer")
        if self.eval_parameters.calc_cpe_in_training:
            module_names.append("reward_model")
        return module_names

    def build_serving_modules(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ):
        """
        `actor_dqn` is the actor module wrapped in the DQN predictor wrapper.
        This helps putting the actor in places where DQN predictor wrapper is expected.
        If the policy is greedy, then this wrapper would work.
        """
        assert isinstance(trainer_module, DiscreteCRRTrainer)
        serving_modules = {
            "default_model": self.build_actor_module(
                trainer_module, normalization_data_map
            ),
            "dqn": self._build_dqn_module(
                trainer_module.q1_network, normalization_data_map
            ),
            "actor_dqn": self._build_dqn_module(
                ActorDQN(trainer_module.actor_network), normalization_data_map
            ),
        }
        if len(self.action_names) == 2:
            serving_modules.update(
                {
                    "binary_difference_scorer": self._build_binary_difference_scorer(
                        ActorDQN(trainer_module.actor_network), normalization_data_map
                    ),
                }
            )
        if self.eval_parameters.calc_cpe_in_training:
            serving_modules.update(
                {
                    "reward_model": self.build_reward_module(
                        trainer_module, normalization_data_map
                    )
                }
            )
        return serving_modules

    def _build_dqn_module(
        self,
        network,
        normalization_data_map: Dict[str, NormalizationData],
    ):
        critic_net_builder = self.critic_net_builder.value
        assert network is not None
        return critic_net_builder.build_serving_module(
            network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )

    def _build_binary_difference_scorer(
        self,
        network,
        normalization_data_map: Dict[str, NormalizationData],
    ):
        critic_net_builder = self.critic_net_builder.value
        assert network is not None
        return critic_net_builder.build_binary_difference_scorer(
            network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )

    # Also, even though the build_serving_module below is directed to
    # discrete_actor_net_builder.py, which returns ActorPredictorWrapper,
    # just like in the continuous_actor_net_builder.py, the outputs of the
    # discrete actor will still be computed differently from those of the
    # continuous actor because during serving, the act() function for the
    # Agent class in gym/agents/agents.py returns
    # self.action_extractor(actor_output), which is created in
    # create_for_env_with_serving_policy, when
    # env.get_serving_action_extractor() is called. During serving,
    # action_extractor calls serving_action_extractor() in env_wrapper.py,
    # which checks the type of action_space during serving time and treats
    # spaces.Discrete differently from spaces.Box (continuous).
    def build_actor_module(
        self,
        trainer_module: DiscreteCRRTrainer,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        return net_builder.build_serving_module(
            trainer_module.actor_network,
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            action_feature_ids=list(range(len(self.action_names))),
        )

    def build_reward_module(
        self,
        trainer_module: DiscreteCRRTrainer,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert trainer_module.reward_network is not None
        net_builder = self.cpe_net_builder.value
        return net_builder.build_serving_module(
            trainer_module.reward_network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
            predictor_wrapper_type=BanditRewardNetPredictorWrapper,
        )


class ActorDQN(ModelBase):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def input_prototype(self):
        return self.actor.input_prototype()

    def forward(self, state):
        return self.actor(state).action
