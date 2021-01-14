#!/usr/bin/env python3

# Note: this file is modeled after td3.py

import logging
from typing import Optional

import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
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
from reagent.parameters import (
    NormalizationData,
    NormalizationParameters,
    EvaluationParameters,
    param_hash,
)
from reagent.training import DiscreteCRRTrainer, CRRTrainerParameters
from reagent.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase
from reagent.workflow.reporters.discrete_crr_reporter import DiscreteCRRReporter

logger = logging.getLogger(__name__)


class ActorPolicyWrapper(Policy):
    """ Actor's forward function is our act """

    def __init__(self, actor_network):
        self.actor_network = actor_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[np.ndarray] = None
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
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: DiscreteActorNetBuilder__Union(
            FullyConnected=DiscreteFullyConnected()
        )
    )

    critic_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
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
        self._actor_network: Optional[ModelBase] = None
        self.rl_parameters = self.trainer_param.rl
        self.action_names = self.trainer_param.actions
        assert (
            len(self.action_names) > 1
        ), f"DiscreteDQNModel needs at least 2 actions. Got {self.action_names}."

    @property
    def action_normalization_data(self) -> NormalizationData:
        return NormalizationData(
            dense_normalization_parameters={
                i: NormalizationParameters(feature_type="DISCRETE_ACTION")
                for i in range(len(self.action_names))
            }
        )

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self) -> DiscreteCRRTrainer:
        actor_net_builder = self.actor_net_builder.value
        # pyre-fixme[16]: `TD3` has no attribute `_actor_network`.
        # pyre-fixme[16]: `TD3` has no attribute `_actor_network`.
        self._actor_network = actor_net_builder.build_actor(
            self.state_normalization_data, self.action_normalization_data
        )

        # The arguments to q_network1 and q_network2 below are modeled after those in discrete_dqn.py
        # The target networks will be created in DiscreteCRRTrainer
        critic_net_builder = self.critic_net_builder.value

        self._q1_network = critic_net_builder.build_q_network(
            self.state_feature_config,
            self.state_normalization_data,
            len(self.action_names),
        )

        q2_network = (
            critic_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_data,
                len(self.action_names),
            )
            if self.trainer_param.double_q_learning
            else None
        )

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.eval_parameters.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(self.metrics_to_score) + 1) * len(
                self.trainer_param.actions
            )

            cpe_net_builder = self.cpe_net_builder.value
            reward_network = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_data,
                num_output_nodes,
            )
            q_network_cpe = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_data,
                num_output_nodes,
            )

            q_network_cpe_target = q_network_cpe.get_target_network()

        trainer = DiscreteCRRTrainer(
            actor_network=self._actor_network,
            q1_network=self._q1_network,
            reward_network=reward_network,
            q2_network=q2_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            evaluation=self.eval_parameters,
            **self.trainer_param.asdict(),
        )
        return trainer

    def create_policy(self, serving: bool) -> Policy:
        """ Create online actor critic policy. """
        if serving:
            return create_predictor_policy_from_model(self.build_serving_module())
        else:
            return ActorPolicyWrapper(self._actor_network)

    def get_reporter(self):
        return DiscreteCRRReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )

    # Note: when using test_gym.py as the entry point, the normalization data
    # is set when the line     normalization = build_normalizer(env)   is executed.
    # The code then calls build_state_normalizer() and build_action_normalizer()
    # in utils.py

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
    def build_serving_module(self) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        assert self._actor_network is not None
        return net_builder.build_serving_module(
            self._actor_network,
            self.state_normalization_data,
            self.action_normalization_data,
        )
