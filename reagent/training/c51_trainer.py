#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import List

import reagent.parameters as rlp
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.tracker import observable
from reagent.parameters import DiscreteActionModelParameters
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.training.training_data_page import TrainingDataPage


@dataclass(frozen=True)
class C51TrainerParameters:
    __hash__ = rlp.param_hash

    actions: List[str] = field(default_factory=list)
    rl: rlp.RLParameters = field(default_factory=rlp.RLParameters)
    double_q_learning: bool = True
    minibatch_size: int = 1024
    minibatches_per_step: int = 1
    num_atoms: int = 51
    qmin: float = -100
    qmax: float = 200
    optimizer: rlp.OptimizerParameters = field(default_factory=rlp.OptimizerParameters)
    evaluation: rlp.EvaluationParameters = field(
        default_factory=rlp.EvaluationParameters
    )

    @classmethod
    def from_discrete_action_model_parameters(
        cls, params: DiscreteActionModelParameters
    ):
        return cls(
            actions=params.actions,
            rl=params.rl,
            double_q_learning=params.rainbow.double_q_learning,
            minibatch_size=params.training.minibatch_size,
            minibatches_per_step=params.training.minibatches_per_step,
            num_atoms=params.rainbow.num_atoms,
            qmin=params.rainbow.qmin,
            qmax=params.rainbow.qmax,
            optimizer=rlp.OptimizerParameters(
                optimizer=params.training.optimizer,
                learning_rate=params.training.learning_rate,
                l2_decay=params.rainbow.c51_l2_decay,
            ),
            evaluation=params.evaluation,
        )


@observable(
    td_loss=torch.Tensor,
    logged_actions=torch.Tensor,
    logged_propensities=torch.Tensor,
    logged_rewards=torch.Tensor,
    model_values=torch.Tensor,
    model_action_idxs=torch.Tensor,
)
class C51Trainer(RLTrainer):
    """
    Implementation of 51 Categorical DQN (C51)

    See https://arxiv.org/abs/1707.06887 for details
    """

    def __init__(
        self,
        q_network,
        q_network_target,
        parameters: C51TrainerParameters,
        use_gpu=False,
        metrics_to_score=None,
        loss_reporter=None,
    ) -> None:
        RLTrainer.__init__(
            self,
            parameters.rl,
            use_gpu=use_gpu,
            metrics_to_score=metrics_to_score,
            actions=parameters.actions,
            loss_reporter=loss_reporter,
        )

        self.double_q_learning = parameters.double_q_learning
        self.minibatch_size = parameters.minibatch_size
        self.minibatches_per_step = parameters.minibatches_per_step or 1
        self._actions = parameters.actions if parameters.actions is not None else []
        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = self._get_optimizer(q_network, parameters.optimizer)
        self.qmin = parameters.qmin
        self.qmax = parameters.qmax
        self.num_atoms = parameters.num_atoms
        self.support = torch.linspace(
            self.qmin, self.qmax, self.num_atoms, device=self.device
        )

        self.reward_boosts = torch.zeros([1, len(self._actions)], device=self.device)
        if parameters.rl.reward_boost is not None:
            # pyre-fixme[16]: Optional type has no attribute `keys`.
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

    @torch.no_grad()
    # pyre-fixme[14]: `train` overrides method defined in `Trainer` inconsistently.
    def train(self, training_batch: rlt.DiscreteDqnInput):
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_discrete_maxq_training_batch()

        rewards = self.boost_rewards(training_batch.reward, training_batch.action)
        discount_tensor = torch.full_like(rewards, self.gamma)
        possible_next_actions_mask = training_batch.possible_next_actions_mask.float()
        possible_actions_mask = training_batch.possible_actions_mask.float()

        self.minibatch += 1
        not_done_mask = training_batch.not_terminal.float()

        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, training_batch.time_diff.float())
        if self.multi_steps is not None:
            assert training_batch.step is not None
            discount_tensor = torch.pow(self.gamma, training_batch.step.float())

        next_dist = self.q_network_target.log_dist(training_batch.next_state).exp()

        if self.maxq_learning:
            # Select distribution corresponding to max valued action
            if self.double_q_learning:
                next_q_values = (
                    self.q_network.log_dist(training_batch.next_state).exp()
                    * self.support
                ).sum(2)
            else:
                next_q_values = (next_dist * self.support).sum(2)

            next_action = self.argmax_with_mask(
                next_q_values, possible_next_actions_mask
            )
            next_dist = next_dist[range(rewards.shape[0]), next_action.reshape(-1)]
        else:
            next_dist = (next_dist * training_batch.next_action.unsqueeze(-1)).sum(1)

        # Build target distribution
        target_Q = rewards + discount_tensor * not_done_mask * self.support

        # Project target distribution back onto support
        # remove support outliers
        target_Q = target_Q.clamp(self.qmin, self.qmax)
        # rescale to indicies
        b = (target_Q - self.qmin) / (self.qmax - self.qmin) * (self.num_atoms - 1.0)
        # pyre-fixme[16]: `Tensor` has no attribute `floor`.
        lower = b.floor()
        # pyre-fixme[16]: `Tensor` has no attribute `ceil`.
        upper = b.ceil()

        # Since index_add_ doesn't work with multiple dimensions
        # we operate on the flattened tensors
        offset = self.num_atoms * torch.arange(
            rewards.shape[0], device=self.device, dtype=torch.long
        ).reshape(-1, 1).repeat(1, self.num_atoms)

        m = torch.zeros_like(next_dist)
        # pyre-fixme[16]: `Tensor` has no attribute `index_add_`.
        m.reshape(-1).index_add_(
            0,
            (lower.long() + offset).reshape(-1),
            (next_dist * (upper - b)).reshape(-1),
        )
        m.reshape(-1).index_add_(
            0,
            (upper.long() + offset).reshape(-1),
            (next_dist * (b - lower)).reshape(-1),
        )

        with torch.enable_grad():
            log_dist = self.q_network.log_dist(training_batch.state)

            # for reporting only
            all_q_values = (log_dist.exp() * self.support).sum(2).detach()

            log_dist = (log_dist * training_batch.action.unsqueeze(-1)).sum(1)

            loss = -(m * log_dist).sum(1).mean()
            loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        model_action_idxs = self.argmax_with_mask(
            all_q_values,
            possible_actions_mask if self.maxq_learning else training_batch.action,
        )

        # pyre-fixme[16]: `C51Trainer` has no attribute `notify_observers`.
        self.notify_observers(
            td_loss=loss,
            logged_actions=torch.argmax(training_batch.action, dim=1, keepdim=True),
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            model_values=all_q_values,
            model_action_idxs=model_action_idxs,
        )

        self.loss_reporter.report(
            td_loss=loss,
            logged_actions=training_batch.action.argmax(dim=1, keepdim=True),
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            model_values=all_q_values,
            model_action_idxs=model_action_idxs,
        )

    @torch.no_grad()
    def internal_prediction(self, input):
        """
        Only used by Gym
        """
        self.q_network.eval()
        q_values = self.q_network(rlt.FeatureData(input))
        q_values = q_values.cpu()
        self.q_network.train()

        return q_values

    @torch.no_grad()
    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts, dim=1, keepdim=True
        )
        return rewards + reward_boosts

    def argmax_with_mask(self, q_values, possible_actions_mask):
        # Set q-values of impossible actions to a very large negative number.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values = q_values + self.ACTION_NOT_POSSIBLE_VAL * (1 - possible_actions_mask)
        return q_values.argmax(1)

    def warm_start_components(self):
        return ["q_network", "q_network_target", "q_network_optimizer"]
