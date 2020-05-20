#!/usr/bin/env python3

import logging

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import (
    GreedyActionSampler,
    SoftmaxActionSampler,
)
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_serving_scorer,
    discrete_qrdqn_scorer,
)
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.quantile_dqn.dueling_quantile import DuelingQuantile
from reagent.net_builder.unions import (
    DiscreteDQNNetBuilder__Union,
    QRDQNNetBuilder__Union,
)
from reagent.parameters import param_hash
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.qrdqn_trainer import QRDQNTrainer, QRDQNTrainerParameters
from reagent.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorUnwrapper as DiscreteDqnPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import DiscreteDqnPredictorUnwrapper


logger = logging.getLogger(__name__)


@dataclass
class DiscreteQRDQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: QRDQNTrainerParameters = field(
        default_factory=QRDQNTrainerParameters
    )
    net_builder: QRDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `DuelingQuantile`.
        default_factory=lambda: QRDQNNetBuilder__Union(
            DuelingQuantile=DuelingQuantile()
        )
    )
    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl
        self.eval_parameters = self.trainer_param.evaluation
        self.action_names = self.trainer_param.actions
        assert len(self.action_names) > 1, "DiscreteQRDQNModel needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    def create_policy(self, serving: bool) -> Policy:
        if serving:
            sampler = GreedyActionSampler()
            scorer = discrete_dqn_serving_scorer(
                DiscreteDqnPredictorUnwrapper(self.build_serving_module())
            )
        else:
            sampler = SoftmaxActionSampler(temperature=self.rl_parameters.temperature)
            # pyre-fixme[16]: `RLTrainer` has no attribute `q_network`.
            scorer = discrete_qrdqn_scorer(self.trainer.q_network)
        return Policy(scorer=scorer, sampler=sampler)

    def build_trainer(self) -> QRDQNTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_normalization_data,
            len(self.action_names),
            num_atoms=self.trainer_param.num_atoms,
        )

        if self.use_gpu:
            q_network = q_network.cuda()

        q_network_target = q_network.get_target_network()

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.trainer_param.evaluation.calc_cpe_in_training:
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

            if self.use_gpu:
                reward_network.cuda()
                q_network_cpe.cuda()

            q_network_cpe_target = q_network_cpe.get_target_network()

        # pyre-fixme[16]: `DiscreteQRDQN` has no attribute `_q_network`.
        self._q_network = q_network
        trainer = QRDQNTrainer(
            q_network,
            q_network_target,
            self.trainer_param,
            self.use_gpu,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            loss_reporter=NoOpLossReporter(),
        )
        return trainer

    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert self._q_network is not None, "_q_network was not initialized"
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            self._q_network,
            self.state_normalization_data,
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
