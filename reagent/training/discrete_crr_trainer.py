#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Note: this files is modeled after td3_trainer.py

import copy
import logging
from typing import List, Tuple

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.parameters import EvaluationParameters, RLParameters
from reagent.training.dqn_trainer_base import DQNTrainerBaseLightning
from torch import distributions as pyd


logger = logging.getLogger(__name__)


class DiscreteCRRTrainer(DQNTrainerBaseLightning):
    """
    Critic Regularized Regression (CRR) algorithm trainer
    as described in https://arxiv.org/abs/2006.15134
    """

    @resolve_defaults
    def __init__(
        self,
        actor_network,
        q1_network,
        reward_network,
        q2_network=None,
        q_network_cpe=None,
        q_network_cpe_target=None,
        metrics_to_score=None,
        evaluation: EvaluationParameters = field(  # noqa: B008
            default_factory=EvaluationParameters
        ),
        # Start CRRTrainerParameters. All parameters above should be
        # in the blacklist for CRRTrainerParameters in parameters.py
        rl: RLParameters = field(default_factory=RLParameters),  # noqa: B008
        double_q_learning: bool = True,
        q_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        actor_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        use_target_actor: bool = False,
        actions: List[str] = field(default_factory=list),  # noqa: B008
        delayed_policy_update: int = 1,
    ) -> None:
        """
        Args:
            actor_network: states -> actions, trained to maximize value
            q1_network: states -> q-value for all actions
            q2_network (optional): double q-learning to stabilize training
                from overestimation bias. The presence of q2_network is specified
                in discrete_crr.py using the config parameter double_q_learning
            rl (optional): an instance of the RLParameter class, which
                defines relevant hyperparameters
            q_network_optimizer (optional): the optimizer class and
                optimizer hyperparameters for the q network(s) optimizer
            actor_network_optimizer (optional): see q_network_optimizer
            use_target_actor (optional): specifies whether target actor is used
            delayed_policy_update (optional): the ratio of q network updates
                to target and policy network updates
        """
        super().__init__(
            rl,
            metrics_to_score=metrics_to_score,
            actions=actions,
            evaluation_parameters=evaluation,
        )
        self._actions = actions
        assert self._actions is not None, "Discrete-action CRR needs action names"

        self.rl_parameters = rl
        self.double_q_learning = double_q_learning

        self.use_target_actor = use_target_actor

        self.q1_network = q1_network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q_network_optimizer = q_network_optimizer

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_target = copy.deepcopy(self.q2_network)

        self.actor_network = actor_network
        self.actor_network_target = copy.deepcopy(self.actor_network)
        self.actor_network_optimizer = actor_network_optimizer

        self.delayed_policy_update = delayed_policy_update

        self.register_buffer("reward_boosts", None)

        self.reward_boosts = torch.zeros([1, len(self._actions)])
        if rl.reward_boost is not None:
            # pyre-fixme[16]: Optional type has no attribute `keys`.
            for k in rl.reward_boost.keys():
                i = self._actions.index(k)
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                self.reward_boosts[0, i] = rl.reward_boost[k]

        self._initialize_cpe(
            reward_network,
            q_network_cpe,
            q_network_cpe_target,
            optimizer=q_network_optimizer,
        )

    @property
    def q_network(self):
        return self.q1_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def get_detached_q_values(self, state) -> Tuple[torch.Tensor, None]:
        # This function is only used in evaluation_data_page.py, in create_from_tensors_dqn(),
        # where two values are expected to be returned from get_detached_q_values(), which
        # is what this function returns in dqn_trainer.py
        q_values = self.q1_network(state)
        return q_values, None

    def configure_optimizers(self):
        optimizers = []

        optimizers.append(
            self.q_network_optimizer.make_optimizer(self.q1_network.parameters())
        )
        if self.q2_network:
            optimizers.append(
                self.q_network_optimizer.make_optimizer(self.q2_network.parameters())
            )
        optimizers.append(
            self.actor_network_optimizer.make_optimizer(self.actor_network.parameters())
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
        target_params = list(self.q1_network_target.parameters())
        source_params = list(self.q1_network.parameters())
        if self.q2_network:
            target_params += list(self.q2_network_target.parameters())
            source_params += list(self.q2_network.parameters())
        target_params += list(self.actor_network_target.parameters())
        source_params += list(self.actor_network.parameters())
        if self.calc_cpe_in_training:
            target_params += list(self.q_network_cpe_target.parameters())
            source_params += list(self.q_network_cpe.parameters())
        optimizers.append(SoftUpdate(target_params, source_params, tau=self.tau))
        return optimizers

    def train_step_gen(self, training_batch: rlt.DiscreteDqnInput, batch_idx: int):
        """
        IMPORTANT: the input action here is preprocessed according to the
        training_batch type, which in this case is DiscreteDqnInput. Hence,
        the preprocessor in the DiscreteDqnInputMaker class in the
        trainer_preprocessor.py is used, which converts acion taken to a
        one-hot representation.
        """
        assert isinstance(training_batch, rlt.DiscreteDqnInput)

        state = training_batch.state
        action = training_batch.action
        next_state = training_batch.next_state
        reward = training_batch.reward
        not_terminal = training_batch.not_terminal

        boosted_rewards = self.boost_rewards(reward, training_batch.action)
        rewards = boosted_rewards

        if self.use_target_actor:
            next_state_actor_output = self.actor_network_target(next_state).action
        else:
            next_state_actor_output = self.actor_network(next_state).action

        next_q_values = self.q1_network_target(next_state)
        next_dist = pyd.Categorical(logits=next_state_actor_output)
        next_V = (next_q_values * next_dist.probs).sum(dim=1, keepdim=True)
        if self.q2_network is not None:
            next_q2_values = self.q2_network_target(next_state)
            next_V2 = (next_q2_values * next_dist.probs).sum(dim=1, keepdim=True)
            next_V = torch.min(next_V, next_V2)

        target_q_value = rewards + self.gamma * next_V * not_terminal.float()

        # Optimize Q1 and Q2
        q1_values = self.q1_network(state)
        # Remember: training_batch.action is in the one-hot format
        logged_action_idxs = torch.argmax(training_batch.action, dim=1, keepdim=True)
        q1 = (q1_values * action).sum(dim=1, keepdim=True)

        q1_loss = F.mse_loss(q1, target_q_value)
        self.reporter.log(
            q1_loss=q1_loss,
            q1_value=q1,
        )
        self.log("td_loss", q1_loss, prog_bar=True)
        yield q1_loss

        if self.q2_network:
            q2_values = self.q2_network(state)
            q2 = (q2_values * action).sum(dim=1, keepdim=True)
            q2_loss = F.mse_loss(q2, target_q_value)
            self.reporter.log(
                q2_loss=q2_loss,
                q2_value=q2,
            )
            yield q2_loss

        all_q_values = self.q1_network(state)  # Q-values of all actions
        all_action_scores = all_q_values.detach()

        # Only update actor and target networks after a fixed number of Q updates
        if batch_idx % self.delayed_policy_update == 0:
            # Note: action_dim (the length of each row of the actor_action
            # matrix obtained below) is assumed to be > 1.
            actor_actions = self.actor_network(state).action
            # dist is the distribution of actions derived from the actor's outputs (logits)
            dist = pyd.Categorical(logits=actor_actions)

            values = (all_q_values * dist.probs).sum(dim=1, keepdim=True)

            advantages = all_q_values - values
            # Note: the above statement subtracts the "values" column vector from
            # every column of the all_q_values matrix, giving us the advantages
            # of every action in the present state

            weight = torch.clamp(
                (advantages * action).sum(dim=1, keepdim=True).exp(), 0, 20.0
            )
            # Note: action space is assumed to be discrete with actions
            # belonging to the set {0, 1, ..., action_dim-1}. Therefore,
            # advantages.gather(1, logged_action_idxs) will select, for each data point
            # (row i of the Advantage matrix "advantages"), the element with index
            # action.float_features[i]

            # Note: dist.logits already gives log(p), which can be verified by
            # comparing dist.probs and dist.logits.
            # https://pytorch.org/docs/master/distributions.html#multinomial
            # states: logits (Tensor) – event log probabilities
            log_pi_b = dist.log_prob(logged_action_idxs.squeeze(1)).unsqueeze(1)

            actor_loss = (-log_pi_b * weight.detach()).mean()

            self.reporter.log(
                actor_loss=actor_loss,
                actor_q1_value=values,
            )
            yield actor_loss
        else:
            # Yielding None prevents the actor and target networks from updating
            yield None
            yield None

        discount_tensor = torch.full_like(rewards, self.gamma)

        yield from self._calculate_cpes(
            training_batch,
            training_batch.state,
            training_batch.next_state,
            all_action_scores,
            next_q_values.detach(),
            logged_action_idxs,
            discount_tensor,
            not_terminal.float(),
        )

        # Do we ever use model_action_idxs computed below?
        model_action_idxs = self.get_max_q_values(
            all_action_scores,
            training_batch.possible_actions_mask
            if self.maxq_learning
            else training_batch.action,
        )[1]

        self.reporter.log(
            logged_actions=logged_action_idxs,
            td_loss=q1_loss,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            model_values=all_action_scores,
            model_action_idxs=model_action_idxs,
        )

        # Use the soft update rule to update the target networks.
        # Note: this yield has to be the last one, since SoftUpdate is the last
        # optimizer added in the configure_optimizers() function.
        result = self.soft_update_result()
        yield result
