#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import reagent.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.parameters import EvaluationParameters, RLParameters
from reagent.training.dqn_trainer_base import DQNTrainerBaseLightning
from reagent.training.imitator_training import get_valid_actions_from_imitator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BCQConfig:
    # 0 = max q-learning, 1 = imitation learning
    drop_threshold: float = 0.1


class DQNTrainer(DQNTrainerBaseLightning):
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
        # Start DQNTrainerParameters
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
        """
        Args:
            q_network: states -> q-value for each action
            q_network_target: copy of q-network for training stability
            reward_network: states -> reward for each action
            q_network_cpe:
            q_network_cpe_target:
            metrics_to_score:
            imitator (optional): The behavior policy, used for BCQ training
            actions: list of action names
            rl: RLParameters
            double_q_learning: boolean flag to use double-q learning
            bcq: a config file for batch-constrained q-learning, defaults to normal
            minibatch_size: samples per minibatch
            minibatches_per_step: minibatch updates per step
            optimizer: q-network optimizer
            evaluation: evaluation params, primarily whether to use CPE in eval or not
        """
        super().__init__(
            rl,
            metrics_to_score=metrics_to_score,
            actions=actions,
            evaluation_parameters=evaluation,
        )
        assert self._actions is not None, "Discrete-action DQN needs action names"
        self.double_q_learning = double_q_learning
        self.minibatch_size = minibatch_size
        self.minibatches_per_step = minibatches_per_step or 1

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = optimizer

        self._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer=optimizer
        )

        self.register_buffer("reward_boosts", None)
        # pyre-fixme[6]: Expected `Sized` for 1st param but got `Optional[List[str]]`.
        self.reward_boosts = torch.zeros([1, len(self._actions)])
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

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.q_network_optimizer.make_optimizer(self.q_network.parameters())
        )
        if self.calc_cpe_in_training:
            optimizers.append(
                self.reward_network_optimizer.make_optimizer(
                    self.reward_network.parameters()
                )
            )
            optimizers.append(
                self.q_network_cpe_optimizer.make_optimizer(
                    self.q_network_cpe.parameters()
                )
            )

        # soft-update
        target_params = list(self.q_network_target.parameters())
        source_params = list(self.q_network.parameters())
        if self.calc_cpe_in_training:
            target_params += list(self.q_network_cpe_target.parameters())
            source_params += list(self.q_network_cpe.parameters())
        optimizers.append(SoftUpdate(target_params, source_params, tau=self.tau))
        return optimizers

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

    def train_step_gen(self, training_batch: rlt.DiscreteDqnInput, batch_idx: int):
        # TODO: calls to _maybe_run_optimizer removed, should be replaced with Trainer parameter
        assert isinstance(training_batch, rlt.DiscreteDqnInput)
        boosted_rewards = self.boost_rewards(
            training_batch.reward, training_batch.action
        )
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
                all_next_q_values,
                all_next_q_values_target,
                possible_next_actions_mask,
            )
        else:
            # SARSA
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                training_batch.next_action,
            )

        filtered_next_q_vals = next_q_values * not_done_mask

        target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        # Get Q-value of action taken
        all_q_values = self.q_network(training_batch.state)
        # pyre-fixme[16]: `DQNTrainer` has no attribute `all_action_scores`.
        self.all_action_scores = all_q_values.detach()
        q_values = torch.sum(all_q_values * training_batch.action, 1, keepdim=True)
        loss = self.q_network_loss(q_values, target_q_values)

        # pyre-fixme[16]: `DQNTrainer` has no attribute `loss`.
        self.loss = loss.detach()
        yield loss

        # Get Q-values of next states, used in computing cpe
        all_next_action_scores = self.q_network(training_batch.next_state).detach()
        logged_action_idxs = torch.argmax(training_batch.action, dim=1, keepdim=True)

        yield from self._calculate_cpes(
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

        # Do we ever use model_action_idxs computed below?
        model_action_idxs = self.get_max_q_values(
            self.all_action_scores,
            possible_actions_mask if self.maxq_learning else training_batch.action,
        )[1]

        self.reporter.log(
            td_loss=self.loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_values=self.all_action_scores,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=model_action_idxs,
        )

        # Use the soft update rule to update target network
        yield self.soft_update_result()
