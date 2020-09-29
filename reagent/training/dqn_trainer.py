#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import reagent.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.core.tracker import observable
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.parameters import EvaluationParameters, RLParameters
from reagent.training.dqn_trainer_base import DQNTrainerBase
from reagent.training.imitator_training import get_valid_actions_from_imitator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BCQConfig:
    # 0 = max q-learning, 1 = imitation learning
    drop_threshold: float = 0.1


@observable(
    td_loss=torch.Tensor,
    reward_loss=torch.Tensor,
    logged_actions=torch.Tensor,
    logged_propensities=torch.Tensor,
    logged_rewards=torch.Tensor,
    model_propensities=torch.Tensor,
    model_rewards=torch.Tensor,
    model_values=torch.Tensor,
    model_action_idxs=torch.Tensor,
)
class DQNTrainer(DQNTrainerBase):
    @resolve_defaults
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        q_network_cpe=None,
        q_network_cpe_target=None,
        metrics_to_score=None,
        imitator=None,
        loss_reporter=None,
        use_gpu: bool = False,
        actions: List[str] = field(default_factory=list),  # noqa: B008
        rl: RLParameters = field(default_factory=RLParameters),  # noqa: B008
        double_q_learning: bool = True,
        bcq: Optional[BCQConfig] = None,
        minibatch_size: int = 1024,
        minibatches_per_step: int = 1,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        evaluation: EvaluationParameters = field(  # noqa: B008
            default_factory=EvaluationParameters
        ),
    ) -> None:
        super().__init__(
            rl,
            use_gpu=use_gpu,
            metrics_to_score=metrics_to_score,
            actions=actions,
            evaluation_parameters=evaluation,
            loss_reporter=loss_reporter,
        )
        assert self._actions is not None, "Discrete-action DQN needs action names"
        self.double_q_learning = double_q_learning
        self.minibatch_size = minibatch_size
        self.minibatches_per_step = minibatches_per_step or 1

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = optimizer.make_optimizer(q_network.parameters())

        self.q_network_soft_update = SoftUpdate(
            self.q_network_target.parameters(), self.q_network.parameters(), self.tau
        )

        self._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer=optimizer
        )

        # pyre-fixme[6]: Expected `Sized` for 1st param but got `Optional[List[str]]`.
        self.reward_boosts = torch.zeros([1, len(self._actions)], device=self.device)
        if rl.reward_boost is not None:
            # pyre-fixme[16]: `Optional` has no attribute `keys`.
            for k in rl.reward_boost.keys():
                # pyre-fixme[16]: `Optional` has no attribute `index`.
                i = self._actions.index(k)
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                self.reward_boosts[0, i] = rl.reward_boost[k]

        # Batch constrained q-learning
        self.bcq = bcq is not None
        if self.bcq:
            assert bcq is not None
            self.bcq_drop_threshold = bcq.drop_threshold
            self.bcq_imitator = imitator

    def warm_start_components(self):
        components = ["q_network", "q_network_target", "q_network_optimizer"]
        if self.reward_network is not None:
            components += [
                "reward_network",
                "reward_network_optimizer",
                "q_network_cpe",
                "q_network_cpe_target",
                "q_network_cpe_optimizer",
            ]
        return components

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def get_detached_q_values(
        self, state
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Gets the q values from the model and target networks """
        q_values = self.q_network(state)
        q_values_target = self.q_network_target(state)
        return q_values, q_values_target

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def train(self, training_batch: rlt.DiscreteDqnInput):
        assert isinstance(training_batch, rlt.DiscreteDqnInput)
        boosted_rewards = self.boost_rewards(
            training_batch.reward, training_batch.action
        )

        self.minibatch += 1
        rewards = boosted_rewards
        discount_tensor = torch.full_like(rewards, self.gamma)
        not_done_mask = training_batch.not_terminal.float()
        assert not_done_mask.dim() == 2

        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, training_batch.time_diff.float())
        if self.multi_steps is not None:
            assert training_batch.step is not None
            # pyre-fixme[16]: `Optional` has no attribute `float`.
            discount_tensor = torch.pow(self.gamma, training_batch.step.float())

        all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
            training_batch.next_state
        )

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            possible_next_actions_mask = (
                training_batch.possible_next_actions_mask.float()
            )
            if self.bcq:
                action_on_policy = get_valid_actions_from_imitator(
                    self.bcq_imitator,
                    training_batch.next_state,
                    self.bcq_drop_threshold,
                )
                possible_next_actions_mask *= action_on_policy
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values, all_next_q_values_target, possible_next_actions_mask
            )
        else:
            # SARSA
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values, all_next_q_values_target, training_batch.next_action
            )

        filtered_next_q_vals = next_q_values * not_done_mask

        target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        with torch.enable_grad():
            # Get Q-value of action taken
            all_q_values = self.q_network(training_batch.state)
            # pyre-fixme[16]: `DQNTrainer` has no attribute `all_action_scores`.
            self.all_action_scores = all_q_values.detach()
            q_values = torch.sum(all_q_values * training_batch.action, 1, keepdim=True)

            loss = self.q_network_loss(q_values, target_q_values)
            # pyre-fixme[16]: `DQNTrainer` has no attribute `loss`.
            self.loss = loss.detach()

            loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_run_optimizer(self.q_network_soft_update, self.minibatches_per_step)

        # Get Q-values of next states, used in computing cpe
        all_next_action_scores = self.q_network(training_batch.next_state).detach()

        logged_action_idxs = torch.argmax(training_batch.action, dim=1, keepdim=True)
        reward_loss, model_rewards, model_propensities = self._calculate_cpes(
            training_batch,
            training_batch.state,
            training_batch.next_state,
            self.all_action_scores,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )

        if self.maxq_learning:
            possible_actions_mask = training_batch.possible_actions_mask

        if self.bcq:
            action_on_policy = get_valid_actions_from_imitator(
                self.bcq_imitator, training_batch.state, self.bcq_drop_threshold
            )
            possible_actions_mask *= action_on_policy

        model_action_idxs = self.get_max_q_values(
            self.all_action_scores,
            possible_actions_mask if self.maxq_learning else training_batch.action,
        )[1]

        # pyre-fixme[16]: `DQNTrainer` has no attribute `notify_observers`.
        self.notify_observers(
            td_loss=self.loss,
            reward_loss=reward_loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            model_values=self.all_action_scores,
            model_action_idxs=model_action_idxs,
        )

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            model_values=self.all_action_scores,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=model_action_idxs,
        )
