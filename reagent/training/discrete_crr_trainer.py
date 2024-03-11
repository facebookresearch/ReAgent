#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

# Note: this files is modeled after td3_trainer.py

import logging
from typing import List, Tuple

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.optimizer import Optimizer__Union, SoftUpdate
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
        actor_network_target,
        q1_network,
        q1_network_target,
        reward_network,
        q2_network=None,
        q2_network_target=None,
        q_network_cpe=None,
        q_network_cpe_target=None,
        metrics_to_score=None,
        evaluation: EvaluationParameters = field(  # noqa: B008
            default_factory=EvaluationParameters
        ),
        # Start CRRTrainerParameters. All parameters above should be
        # in the blocklist for CRRTrainerParameters in parameters.py
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
        beta: float = 1.0,
        entropy_coeff: float = 0.0,
        clip_limit: float = 10.0,
        max_weight: float = 20.0,
    ) -> None:
        """
        Args:
            actor_network: states -> actions, trained to maximize value
            actor_network_target: copy of actor network for training stability
            q1_network: states -> q-value for all actions
            q1_network_target: copy of q-network for training stability
            q2_network (optional): double q-learning to stabilize training
                from overestimation bias. The presence of q2_network is specified
                in discrete_crr.py using the config parameter double_q_learning
            q2_network_target (optional): copy of q-network for training stability
            rl (optional): an instance of the RLParameter class, which
                defines relevant hyperparameters
            q_network_optimizer (optional): the optimizer class and
                optimizer hyperparameters for the q network(s) optimizer
            actor_network_optimizer (optional): see q_network_optimizer
            use_target_actor (optional): specifies whether target actor is used
            delayed_policy_update (optional): the ratio of q network updates
                to target and policy network updates
            beta: coefficient for KL-divergence policy constaint regularization of CRR
                see eq(5) in https://arxiv.org/pdf/2006.15134.pdf. With large beta, the output
                policy of CRR can not leaves too far away from the logged policy

            entropy_coeff: coefficient for entropy regularization
            clip_limit: threshold for importance sampling when compute entropy
                regularization using offline samples
            max_weight: the maximum possible action weight in the actor loss

            Explaination of entropy regularization:
            Entropy regularization punishes deterministic policy and encourages
            "unifom" policy. Entropy regularized MDP can be viewed as add the term
            (-entropy_coeff * pi_ratio * log_pi_b) to each reward. For detailed
            formulation of entropy regularized please see eq.(9) & eq.(10) in
            https://arxiv.org/pdf/2007.06558.pdf
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
        self.q1_network_target = q1_network_target
        self.q_network_optimizer = q_network_optimizer

        self.q2_network = q2_network
        if self.q2_network is not None:
            assert (
                q2_network_target is not None
            ), "q2_network provided without a target network"
            self.q2_network_target = q2_network_target

        self.actor_network = actor_network
        self.actor_network_target = actor_network_target
        self.actor_network_optimizer = actor_network_optimizer

        self.delayed_policy_update = delayed_policy_update

        self._initialize_cpe(
            reward_network,
            q_network_cpe,
            q_network_cpe_target,
            optimizer=q_network_optimizer,
        )
        self.beta = beta
        self.entropy_coeff = entropy_coeff
        self.clip_limit = clip_limit
        self.max_weight = max_weight

    @property
    def q_network(self):
        return self.q1_network

    @torch.no_grad()
    def get_detached_model_outputs(self, state) -> Tuple[torch.Tensor, None]:
        # This function is only used in evaluation_data_page.py, in create_from_tensors_dqn(),
        # in order to compute model propensities. The definition of this function in
        # dqn_trainer.py returns two values, and so we also return two values here, for
        # consistency.
        action_scores = self.actor_network(state).action
        return action_scores, None

    def configure_optimizers(self):
        optimizers = []
        target_params = list(self.q1_network_target.parameters())
        source_params = list(self.q1_network.parameters())

        optimizers.append(
            self.q_network_optimizer.make_optimizer_scheduler(
                self.q1_network.parameters()
            )
        )
        if self.q2_network:
            target_params += list(self.q2_network_target.parameters())
            source_params += list(self.q2_network.parameters())
            optimizers.append(
                self.q_network_optimizer.make_optimizer_scheduler(
                    self.q2_network.parameters()
                )
            )

        target_params += list(self.actor_network_target.parameters())
        source_params += list(self.actor_network.parameters())
        optimizers.append(
            self.actor_network_optimizer.make_optimizer_scheduler(
                self.actor_network.parameters()
            )
        )

        if self.calc_cpe_in_training:
            (
                cpe_target_params,
                cpe_source_params,
                cpe_optimizers,
            ) = self._configure_cpe_optimizers()
            target_params += cpe_target_params
            source_params += cpe_source_params
            optimizers += cpe_optimizers

        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def compute_target_q_values(self, next_state, rewards, not_terminal, next_q_values):
        if self.use_target_actor:
            next_state_actor_output = self.actor_network_target(next_state).action
        else:
            next_state_actor_output = self.actor_network(next_state).action

        next_dist = pyd.Categorical(logits=next_state_actor_output)
        next_V = (next_q_values * next_dist.probs).sum(dim=1, keepdim=True)
        if self.q2_network is not None:
            next_q2_values = self.q2_network_target(next_state)
            next_V2 = (next_q2_values * next_dist.probs).sum(dim=1, keepdim=True)
            next_V = torch.min(next_V, next_V2)

        target_q_values = rewards + self.gamma * next_V * not_terminal.float()
        return target_q_values

    def compute_td_loss(self, q_network, state, action, target_q_values):
        all_q_values = q_network(state)
        q_values = (all_q_values * action).sum(dim=1, keepdim=True)
        q_loss = F.mse_loss(q_values, target_q_values)
        return q_loss

    def compute_actor_loss(
        self, batch_idx, action, logged_action_probs, all_q_values, all_action_scores
    ):
        # Only update actor network after a fixed number of Q updates
        if batch_idx % self.delayed_policy_update != 0:
            # Yielding None prevents the actor network from updating
            actor_loss = None
            return (actor_loss, actor_loss)

        # dist is the distribution of actions derived from the actor's outputs (logits)
        dist = pyd.Categorical(logits=all_action_scores)
        # Note: D = dist.probs is equivalent to:
        # e_x = torch.exp(actor_actions)
        # D = e_x / e_x.sum(dim=1, keepdim=True)
        # That is, dist gives a softmax distribution over actor's outputs

        # values is the vector of state values in this batch
        values = (all_q_values * dist.probs).sum(dim=1, keepdim=True)

        advantages = all_q_values - values
        # Note: the above statement subtracts the "values" column vector from
        # every column of the all_q_values matrix, giving us the advantages
        # of every action in the present state

        weight = torch.clamp(
            ((1 / self.beta) * (advantages * action).sum(dim=1, keepdim=True)).exp(),
            0,
            self.max_weight,
        )
        # Remember: training_batch.action is in the one-hot format
        logged_action_idxs = torch.argmax(action, dim=1, keepdim=True)

        # Note: action space is assumed to be discrete with actions
        # belonging to the set {0, 1, ..., action_dim-1}. Therefore,
        # advantages.gather(1, logged_action_idxs) will select, for each data point
        # (row i of the Advantage matrix "advantages"), the element with index
        # action.float_features[i]

        # Note: dist.logits already gives log(p), which can be verified by
        # comparing dist.probs and dist.logits.
        # https://pytorch.org/docs/master/distributions.html#multinomial
        # states: logits (Tensor) – event log probabilities

        # log_pi_b is the log of the probability assigned by the
        # actor (abbreviated as pi) to the actions of the behavioral (b) policy
        log_pi_b = dist.log_prob(logged_action_idxs.squeeze(1)).unsqueeze(1)

        # entropy regularization
        pi_t = (dist.probs * action).sum(dim=1, keepdim=True)

        if self.entropy_coeff > 0:
            pi_b = logged_action_probs.view(pi_t.shape)
            assert torch.min(pi_b) > 0, "Logged action probability <= 0"
            pi_ratio = torch.clip(pi_t / pi_b, min=1e-4, max=self.clip_limit)
            entropy = (pi_ratio * log_pi_b).mean()
        else:
            # dummy value
            entropy = 0

        # Note: the CRR loss for each datapoint (and the magnitude of the corresponding
        # parameter update) is proportional to log_pi_b * weight. Therefore, as mentioned
        # at the top of Section 3.2, the actor on the one hand has incentive to assign
        # larger probabilities to the actions observed in the dataset (so as to reduce
        # the magnitude of log_pi_b), but on the other hand it gives preference to doing
        # this on datapoints where weight is large (i.e., those points on which the
        # Q-value of the observed action is large).
        actor_loss_without_reg = (-log_pi_b * weight.detach()).mean()
        actor_loss = (-log_pi_b * weight.detach()).mean() + self.entropy_coeff * entropy
        return actor_loss_without_reg, actor_loss

    def train_step_gen(self, training_batch: rlt.DiscreteDqnInput, batch_idx: int):
        """
        IMPORTANT: the input action here is preprocessed according to the
        training_batch type, which in this case is DiscreteDqnInput. Hence,
        the preprocessor in the DiscreteDqnInputMaker class in the
        trainer_preprocessor.py is used, which converts acion taken to a
        one-hot representation.
        """
        self._check_input(training_batch)

        state = training_batch.state
        action = training_batch.action
        next_state = training_batch.next_state
        not_terminal = training_batch.not_terminal
        rewards = self.boost_rewards(training_batch.reward, training_batch.action)

        # Remember: training_batch.action is in the one-hot format
        logged_action_idxs = torch.argmax(action, dim=1, keepdim=True)
        discount_tensor = torch.full_like(rewards, self.gamma)

        next_q_values = self.q1_network_target(next_state)
        target_q_values = self.compute_target_q_values(
            next_state, rewards, not_terminal, next_q_values
        )
        q1_loss = self.compute_td_loss(self.q1_network, state, action, target_q_values)

        # Show td_loss on the progress bar and in tensorboard graphs:
        self.log(
            "td_loss", q1_loss, prog_bar=True, batch_size=training_batch.batch_size()
        )
        yield q1_loss

        if self.q2_network:
            q2_loss = self.compute_td_loss(
                self.q2_network, state, action, target_q_values
            )
            yield q2_loss

        all_q_values = self.q1_network(state)  # Q-values of all actions

        # Note: action_dim (the length of each row of the actor_action
        # matrix obtained below) is assumed to be > 1.
        all_action_scores = self.actor_network(state).action
        logged_action_probs = training_batch.extras.action_probability

        actor_loss_without_reg, actor_loss = self.compute_actor_loss(
            batch_idx, action, logged_action_probs, all_q_values, all_action_scores
        )
        # self.reporter.log(
        #     actor_loss=actor_loss,
        #     actor_q1_value=actor_q1_values,
        # )

        # Show actor_loss on the progress bar and also in Tensorboard graphs
        self.log(
            "actor_loss_without_reg",
            actor_loss_without_reg,
            prog_bar=True,
            batch_size=training_batch.batch_size(),
        )
        self.log(
            "actor_loss",
            actor_loss,
            prog_bar=True,
            batch_size=training_batch.batch_size(),
        )
        yield actor_loss

        yield from self._calculate_cpes(
            training_batch,
            state,
            next_state,
            all_action_scores,
            next_q_values.detach(),
            logged_action_idxs,
            discount_tensor,
            not_terminal.float(),
        )

        # TODO: rename underlying function to get_max_possible_values_and_idxs
        model_action_idxs = self.get_max_q_values(
            all_action_scores,
            training_batch.possible_actions_mask if self.maxq_learning else action,
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

    def validation_step(self, batch, batch_idx):
        # As explained in the comments to the validation_step function in
        # pytorch_lightning/core/lightning.py, this function operates on a
        # single batch of data from the validation set. For example:
        # val_outs = []
        # for val_batch in val_data:
        #     out = validation_step(val_batch)
        #     val_outs.append(out)
        # validation_epoch_end(val_outs)
        # Note: the relevant validation_epoch_end() function is defined in dqn_trainer_base.py

        # RETURN ARGS:
        # The super() call at the end of this function calls the function with the same name
        # in dqn_trainer_base.py, which returns a EvaluationDataPage for data in that batch.
        # In other words, the validation_epoch_end() function will take a list of validation
        # EvaluationDataPages.

        if isinstance(batch, dict):
            batch = rlt.DiscreteDqnInput.from_dict(batch)

        # validation data
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        not_terminal = batch.not_terminal
        rewards = self.boost_rewards(batch.reward, action)

        # intermediate values
        next_q_values = self.q1_network_target(next_state)
        target_q_values = self.compute_target_q_values(
            next_state, rewards, not_terminal, next_q_values
        )
        all_q_values = self.q1_network(state)
        all_action_scores = self.actor_network(state).action
        logged_action_probs = batch.extras.action_probability

        # loss to log
        actor_loss_without_reg, actor_loss = self.compute_actor_loss(
            batch_idx, action, logged_action_probs, all_q_values, all_action_scores
        )
        td_loss = self.compute_td_loss(self.q1_network, state, action, target_q_values)

        self.log(
            "eval_actor_loss_without_reg",
            actor_loss_without_reg,
            batch_size=batch.batch_size(),
        )
        self.log("eval_actor_loss", actor_loss, batch_size=batch.batch_size())
        self.log("eval_td_loss", td_loss, batch_size=batch.batch_size())

        return super().validation_step(batch, batch_idx)
