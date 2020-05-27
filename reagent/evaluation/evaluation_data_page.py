#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import NamedTuple, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from reagent import types as rlt
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.torch_utils import masked_softmax
from reagent.training.dqn_trainer import DQNTrainer
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class EvaluationDataPage(NamedTuple):
    mdp_id: Optional[torch.Tensor]
    sequence_number: Optional[torch.Tensor]
    logged_propensities: torch.Tensor
    logged_rewards: torch.Tensor
    action_mask: torch.Tensor
    model_propensities: torch.Tensor
    model_rewards: torch.Tensor
    model_rewards_for_logged_action: torch.Tensor
    model_values: Optional[torch.Tensor] = None
    model_values_for_logged_action: Optional[torch.Tensor] = None
    possible_actions_mask: Optional[torch.Tensor] = None
    optimal_q_values: Optional[torch.Tensor] = None
    eval_action_idxs: Optional[torch.Tensor] = None
    logged_values: Optional[torch.Tensor] = None
    logged_metrics: Optional[torch.Tensor] = None
    logged_metrics_values: Optional[torch.Tensor] = None
    model_metrics: Optional[torch.Tensor] = None
    model_metrics_for_logged_action: Optional[torch.Tensor] = None
    model_metrics_values: Optional[torch.Tensor] = None
    model_metrics_values_for_logged_action: Optional[torch.Tensor] = None
    possible_actions_state_concat: Optional[torch.Tensor] = None
    contexts: Optional[torch.Tensor] = None

    @classmethod
    def create_from_training_batch(
        cls,
        tdb: rlt.PreprocessedTrainingBatch,
        trainer: Trainer,
        reward_network: Optional[nn.Module] = None,
    ):
        if isinstance(tdb, rlt.DiscreteDqnInput):
            # pyre-fixme[22]: The cast is redundant.
            discrete_training_input = cast(rlt.DiscreteDqnInput, tdb)

            return EvaluationDataPage.create_from_tensors_dqn(
                # pyre-fixme[6]: Expected `DQNTrainer` for 1st param but got `Trainer`.
                trainer,
                tdb.extras.mdp_id,
                tdb.extras.sequence_number,
                discrete_training_input.state,
                discrete_training_input.action,
                tdb.extras.action_probability,
                discrete_training_input.reward,
                discrete_training_input.possible_actions_mask,
                metrics=tdb.extras.metrics,
            )
        elif isinstance(tdb, rlt.ParametricDqnInput):
            return EvaluationDataPage.create_from_tensors_parametric_dqn(
                # pyre-fixme[6]: Expected `ParametricDQNTrainer` for 1st param but
                #  got `Trainer`.
                trainer,
                # pyre-fixme[16]: `Optional` has no attribute `mdp_id`.
                tdb.extras.mdp_id,
                # pyre-fixme[16]: `Optional` has no attribute `sequence_number`.
                tdb.extras.sequence_number,
                tdb.state,
                tdb.action,
                # pyre-fixme[16]: `Optional` has no attribute `action_probability`.
                tdb.extras.action_probability,
                tdb.reward,
                tdb.possible_actions_mask,
                tdb.possible_actions,
                # pyre-fixme[16]: `Optional` has no attribute `max_num_actions`.
                tdb.extras.max_num_actions,
                # pyre-fixme[16]: `Optional` has no attribute `metrics`.
                metrics=tdb.extras.metrics,
            )
        else:
            raise NotImplementedError(
                f"training_input type: {type(tdb.training_input)}"
            )

    @classmethod
    @torch.no_grad()
    def create_from_tensors_seq2slate(
        cls,
        seq2slate_net: Seq2SlateTransformerNet,
        reward_network: nn.Module,
        training_input: rlt.PreprocessedRankingInput,
        eval_greedy: bool,
        mdp_ids: Optional[torch.Tensor] = None,
        sequence_numbers: Optional[torch.Tensor] = None,
    ):
        """
        :param eval_greedy: If True, evaluate the greedy policy which
        always picks the most probable output sequence. If False, evaluate
         the stochastic ranking policy.
        """
        assert (
            training_input.slate_reward is not None
            and training_input.tgt_out_probs is not None
            and training_input.tgt_out_idx is not None
            and training_input.tgt_out_seq is not None
        )
        (
            batch_size,
            tgt_seq_len,
            candidate_dim,
            # pyre-fixme[16]: `Optional` has no attribute `float_features`.
        ) = training_input.tgt_out_seq.float_features.shape
        device = training_input.state.float_features.device

        rank_output = seq2slate_net(
            training_input, Seq2SlateMode.RANK_MODE, greedy=True
        )
        assert rank_output.ranked_tgt_out_idx is not None
        if eval_greedy:
            model_propensities = torch.ones(batch_size, 1, device=device)
            action_mask = torch.all(
                # pyre-fixme[6]: Expected `int` for 1st param but got
                #  `Optional[torch.Tensor]`.
                (training_input.tgt_out_idx - 2)
                == (rank_output.ranked_tgt_out_idx - 2),
                dim=1,
                keepdim=True,
            ).float()
        else:
            # Fully evaluating a non-greedy ranking model is too expensive because
            # we would need to compute propensities of all possible output sequences.
            # Here we only compute the propensity of the output sequences in the data.
            # As a result, we can still get a true IPS estimation but not correct
            # direct method / doubly-robust.
            model_propensities = torch.exp(
                seq2slate_net(
                    training_input, Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
                ).log_probs
            )
            action_mask = torch.ones(batch_size, 1, device=device).float()

        model_rewards_for_logged_action = reward_network(
            training_input.state.float_features,
            training_input.src_seq.float_features,
            training_input.tgt_out_seq.float_features,
            training_input.src_src_mask,
            training_input.tgt_out_idx,
        ).reshape(-1, 1)

        ranked_tgt_out_seq = training_input.src_seq.float_features[
            # pyre-fixme[16]: `Tensor` has no attribute `repeat_interleave`.
            torch.arange(batch_size, device=device).repeat_interleave(tgt_seq_len),
            rank_output.ranked_tgt_out_idx.flatten() - 2,
        ].reshape(batch_size, tgt_seq_len, candidate_dim)
        # model_rewards refers to predicted rewards for the slate generated
        # greedily by the ranking model. It would be too expensive to
        # compute model_rewards for all possible slates
        model_rewards = reward_network(
            training_input.state.float_features,
            training_input.src_seq.float_features,
            ranked_tgt_out_seq,
            training_input.src_src_mask,
            rank_output.ranked_tgt_out_idx,
        ).reshape(-1, 1)
        # pyre-fixme[16]: `Optional` has no attribute `reshape`.
        logged_rewards = training_input.slate_reward.reshape(-1, 1)
        logged_propensities = training_input.tgt_out_probs.reshape(-1, 1)
        return cls(
            mdp_id=mdp_ids,
            sequence_number=sequence_numbers,
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            action_mask=action_mask,
            logged_rewards=logged_rewards,
            model_rewards_for_logged_action=model_rewards_for_logged_action,
            logged_propensities=logged_propensities,
        )

    @classmethod
    @torch.no_grad()
    def create_from_tensors_parametric_dqn(
        cls,
        trainer: ParametricDQNTrainer,
        mdp_ids: torch.Tensor,
        sequence_numbers: torch.Tensor,
        states: rlt.FeatureData,
        actions: rlt.FeatureData,
        propensities: torch.Tensor,
        rewards: torch.Tensor,
        possible_actions_mask: torch.Tensor,
        possible_actions: rlt.FeatureData,
        max_num_actions: int,
        metrics: Optional[torch.Tensor] = None,
    ):
        old_q_train_state = trainer.q_network.training
        old_reward_train_state = trainer.reward_network.training
        trainer.q_network.train(False)
        trainer.reward_network.train(False)

        tiled_state = states.float_features.repeat(1, max_num_actions).reshape(
            -1, states.float_features.shape[1]
        )
        assert possible_actions is not None
        # Get Q-value of action taken
        possible_actions_state_concat = (rlt.FeatureData(tiled_state), possible_actions)

        # FIXME: model_values, model_values_for_logged_action, and model_metrics_values
        # should be calculated using q_network_cpe (as in discrete dqn).
        # q_network_cpe has not been added in parametric dqn yet.
        model_values = trainer.q_network(*possible_actions_state_concat)
        optimal_q_values, _ = trainer.get_detached_q_values(
            *possible_actions_state_concat
        )
        eval_action_idxs = None

        assert (
            model_values.shape[1] == 1
            and model_values.shape[0]
            == possible_actions_mask.shape[0] * possible_actions_mask.shape[1]
        ), (
            "Invalid shapes: "
            + str(model_values.shape)
            + " != "
            + str(possible_actions_mask.shape)
        )
        model_values = model_values.reshape(possible_actions_mask.shape)
        optimal_q_values = optimal_q_values.reshape(possible_actions_mask.shape)
        model_propensities = masked_softmax(
            optimal_q_values, possible_actions_mask, trainer.rl_temperature
        )

        rewards_and_metric_rewards = trainer.reward_network(
            *possible_actions_state_concat
        )
        model_rewards = rewards_and_metric_rewards[:, :1]
        assert (
            model_rewards.shape[0] * model_rewards.shape[1]
            == possible_actions_mask.shape[0] * possible_actions_mask.shape[1]
        ), (
            "Invalid shapes: "
            + str(model_rewards.shape)
            + " != "
            + str(possible_actions_mask.shape)
        )
        model_rewards = model_rewards.reshape(possible_actions_mask.shape)

        model_metrics = rewards_and_metric_rewards[:, 1:]
        model_metrics = model_metrics.reshape(possible_actions_mask.shape[0], -1)

        model_values_for_logged_action = trainer.q_network(states, actions)
        model_rewards_and_metrics_for_logged_action = trainer.reward_network(
            states, actions
        )
        model_rewards_for_logged_action = model_rewards_and_metrics_for_logged_action[
            :, :1
        ]

        action_dim = possible_actions.float_features.shape[1]
        action_mask = torch.all(
            possible_actions.float_features.view(-1, max_num_actions, action_dim)
            == actions.float_features.unsqueeze(dim=1),
            dim=2,
        ).float()
        assert torch.all(action_mask.sum(dim=1) == 1)
        num_metrics = model_metrics.shape[1] // max_num_actions

        model_metrics_values = None
        model_metrics_for_logged_action = None
        model_metrics_values_for_logged_action = None
        if num_metrics > 0:
            # FIXME: calculate model_metrics_values when q_network_cpe is added
            # to parametric dqn
            model_metrics_values = model_values.repeat(1, num_metrics)

        trainer.q_network.train(old_q_train_state)
        trainer.reward_network.train(old_reward_train_state)

        return cls(
            mdp_id=mdp_ids,
            sequence_number=sequence_numbers,
            logged_propensities=propensities,
            logged_rewards=rewards,
            action_mask=action_mask,
            model_rewards=model_rewards,
            model_rewards_for_logged_action=model_rewards_for_logged_action,
            model_values=model_values,
            model_values_for_logged_action=model_values_for_logged_action,
            model_metrics_values=model_metrics_values,
            model_metrics_values_for_logged_action=model_metrics_values_for_logged_action,
            model_propensities=model_propensities,
            logged_metrics=metrics,
            model_metrics=model_metrics,
            model_metrics_for_logged_action=model_metrics_for_logged_action,
            # Will compute later
            logged_values=None,
            logged_metrics_values=None,
            possible_actions_mask=possible_actions_mask,
            optimal_q_values=optimal_q_values,
            eval_action_idxs=eval_action_idxs,
        )

    @classmethod
    @torch.no_grad()
    def create_from_tensors_dqn(
        cls,
        trainer: DQNTrainer,
        mdp_ids: torch.Tensor,
        sequence_numbers: torch.Tensor,
        states: rlt.FeatureData,
        actions: rlt.FeatureData,
        propensities: torch.Tensor,
        rewards: torch.Tensor,
        possible_actions_mask: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ):
        old_q_train_state = trainer.q_network.training
        # pyre-fixme[16]: `DQNTrainer` has no attribute `reward_network`.
        old_reward_train_state = trainer.reward_network.training
        # pyre-fixme[16]: `DQNTrainer` has no attribute `q_network_cpe`.
        old_q_cpe_train_state = trainer.q_network_cpe.training
        trainer.q_network.train(False)
        trainer.reward_network.train(False)
        trainer.q_network_cpe.train(False)

        num_actions = trainer.num_actions
        action_mask = actions.float()

        # pyre-fixme[6]: Expected `torch.Tensor` for 2nd positional only parameter
        rewards = trainer.boost_rewards(rewards, actions)
        model_values = trainer.q_network_cpe(states)[:, 0:num_actions]
        optimal_q_values, _ = trainer.get_detached_q_values(states)
        eval_action_idxs = trainer.get_max_q_values(
            optimal_q_values, possible_actions_mask
        )[1]
        model_propensities = masked_softmax(
            optimal_q_values, possible_actions_mask, trainer.rl_temperature
        )
        assert model_values.shape == actions.shape, (
            "Invalid shape: " + str(model_values.shape) + " != " + str(actions.shape)
        )
        assert model_values.shape == possible_actions_mask.shape, (
            "Invalid shape: "
            + str(model_values.shape)
            + " != "
            + str(possible_actions_mask.shape)
        )
        model_values_for_logged_action = torch.sum(
            model_values * action_mask, dim=1, keepdim=True
        )

        rewards_and_metric_rewards = trainer.reward_network(states)

        # In case we reuse the modular for Q-network
        if hasattr(rewards_and_metric_rewards, "q_values"):
            rewards_and_metric_rewards = rewards_and_metric_rewards

        model_rewards = rewards_and_metric_rewards[:, 0:num_actions]
        assert model_rewards.shape == actions.shape, (
            "Invalid shape: " + str(model_rewards.shape) + " != " + str(actions.shape)
        )
        model_rewards_for_logged_action = torch.sum(
            model_rewards * action_mask, dim=1, keepdim=True
        )

        model_metrics = rewards_and_metric_rewards[:, num_actions:]

        assert model_metrics.shape[1] % num_actions == 0, (
            "Invalid metrics shape: "
            + str(model_metrics.shape)
            + " "
            + str(num_actions)
        )
        num_metrics = model_metrics.shape[1] // num_actions

        if num_metrics == 0:
            model_metrics_values = None
            model_metrics_for_logged_action = None
            model_metrics_values_for_logged_action = None
        else:
            model_metrics_values = trainer.q_network_cpe(states)
            # Backward compatility
            if hasattr(model_metrics_values, "q_values"):
                model_metrics_values = model_metrics_values
            model_metrics_values = model_metrics_values[:, num_actions:]
            assert model_metrics_values.shape[1] == num_actions * num_metrics, (
                "Invalid shape: "
                + str(model_metrics_values.shape[1])
                + " != "
                + str(actions.shape[1] * num_metrics)
            )

            model_metrics_for_logged_action_list = []
            model_metrics_values_for_logged_action_list = []
            for metric_index in range(num_metrics):
                metric_start = metric_index * num_actions
                metric_end = (metric_index + 1) * num_actions
                model_metrics_for_logged_action_list.append(
                    torch.sum(
                        model_metrics[:, metric_start:metric_end] * action_mask,
                        dim=1,
                        keepdim=True,
                    )
                )

                model_metrics_values_for_logged_action_list.append(
                    torch.sum(
                        model_metrics_values[:, metric_start:metric_end] * action_mask,
                        dim=1,
                        keepdim=True,
                    )
                )
            model_metrics_for_logged_action = torch.cat(
                model_metrics_for_logged_action_list, dim=1
            )
            model_metrics_values_for_logged_action = torch.cat(
                model_metrics_values_for_logged_action_list, dim=1
            )

        trainer.q_network_cpe.train(old_q_cpe_train_state)
        trainer.q_network.train(old_q_train_state)
        trainer.reward_network.train(old_reward_train_state)

        return cls(
            mdp_id=mdp_ids,
            sequence_number=sequence_numbers,
            logged_propensities=propensities,
            logged_rewards=rewards,
            action_mask=action_mask,
            model_rewards=model_rewards,
            model_rewards_for_logged_action=model_rewards_for_logged_action,
            model_values=model_values,
            model_values_for_logged_action=model_values_for_logged_action,
            model_metrics_values=model_metrics_values,
            model_metrics_values_for_logged_action=model_metrics_values_for_logged_action,
            model_propensities=model_propensities,
            logged_metrics=metrics,
            model_metrics=model_metrics,
            model_metrics_for_logged_action=model_metrics_for_logged_action,
            # Will compute later
            logged_values=None,
            logged_metrics_values=None,
            possible_actions_mask=possible_actions_mask,
            optimal_q_values=optimal_q_values,
            eval_action_idxs=eval_action_idxs,
        )

    def append(self, edp):
        new_edp = {}
        for x in EvaluationDataPage._fields:
            t = getattr(self, x)
            other_t = getattr(edp, x)
            assert int(t is not None) + int(other_t is not None) != 1, (
                "Tried to append when a tensor existed in one training page but not the other: "
                + x
            )
            if other_t is not None:
                if isinstance(t, torch.Tensor):
                    new_edp[x] = torch.cat((t, other_t), dim=0)
                elif isinstance(t, np.ndarray):
                    new_edp[x] = np.concatenate((t, other_t), axis=0)
                else:
                    raise Exception("Invalid type in training data page")
            else:
                new_edp[x] = None
        return EvaluationDataPage(**new_edp)

    def sort(self):
        idxs = []
        for i, (mdp_id, seq_num) in enumerate(zip(self.mdp_id, self.sequence_number)):
            idxs.append((mdp_id, int(seq_num), i))
        sorted_idxs = [i for _mdp_id, _seq_num, i in sorted(idxs)]
        new_edp = {}
        for x in EvaluationDataPage._fields:
            t = getattr(self, x)
            new_edp[x] = t[sorted_idxs] if t is not None else None

        return EvaluationDataPage(**new_edp)

    def compute_values(self, gamma: float):
        assert self.mdp_id is not None and self.sequence_number is not None
        logged_values = EvaluationDataPage.compute_values_for_mdps(
            self.logged_rewards, self.mdp_id, self.sequence_number, gamma
        )
        if self.logged_metrics is not None:
            logged_metrics_values: Optional[
                torch.Tensor
            ] = EvaluationDataPage.compute_values_for_mdps(
                self.logged_metrics, self.mdp_id, self.sequence_number, gamma
            )
        else:
            logged_metrics_values = None
        return self._replace(
            logged_values=logged_values, logged_metrics_values=logged_metrics_values
        )

    @staticmethod
    def compute_values_for_mdps(
        rewards: torch.Tensor,
        mdp_ids: torch.Tensor,
        sequence_numbers: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        values = rewards.clone()

        for x in range(len(values) - 2, -1, -1):
            if mdp_ids[x] != mdp_ids[x + 1]:
                # Value = reward for new mdp_id
                continue
            values[x, 0] += values[x + 1, 0] * math.pow(
                gamma, float(sequence_numbers[x + 1, 0] - sequence_numbers[x, 0])
            )

        return values

    def validate(self):
        assert len(self.logged_propensities.shape) == 2
        assert len(self.logged_rewards.shape) == 2
        assert len(self.logged_values.shape) == 2
        assert len(self.model_propensities.shape) == 2
        assert len(self.model_rewards.shape) == 2
        assert len(self.model_values.shape) == 2

        assert self.logged_propensities.shape[1] == 1
        assert self.logged_rewards.shape[1] == 1
        assert self.logged_values.shape[1] == 1
        num_actions = self.model_propensities.shape[1]
        assert self.model_rewards.shape[1] == num_actions
        assert self.model_values.shape[1] == num_actions

        assert self.action_mask.shape == self.model_propensities.shape

        if self.logged_metrics is not None:
            assert len(self.logged_metrics.shape) == 2
            assert len(self.logged_metrics_values.shape) == 2
            assert len(self.model_metrics.shape) == 2
            assert len(self.model_metrics_values.shape) == 2

            num_metrics = self.logged_metrics.shape[1]
            assert self.logged_metrics_values.shape[1] == num_metrics, (
                "Invalid shape: "
                + str(self.logged_metrics_values.shape)
                + " != "
                + str(num_metrics)
            )
            assert self.model_metrics.shape[1] == num_metrics * num_actions, (
                "Invalid shape: "
                + str(self.model_metrics.shape)
                + " != "
                + str(num_metrics * num_actions)
            )
            assert self.model_metrics_values.shape[1] == num_metrics * num_actions

        minibatch_size = self.logged_propensities.shape[0]
        logger.info("EvaluationDataPage minibatch size: {}".format(minibatch_size))
        assert minibatch_size == self.logged_rewards.shape[0]
        assert minibatch_size == self.logged_values.shape[0]
        assert minibatch_size == self.model_propensities.shape[0]
        assert minibatch_size == self.model_rewards.shape[0]
        assert minibatch_size == self.model_values.shape[0]
        if self.logged_metrics is not None:
            assert minibatch_size == self.logged_metrics.shape[0]
            assert minibatch_size == self.logged_metrics_values.shape[0]
            assert minibatch_size == self.model_metrics.shape[0]
            assert minibatch_size == self.model_metrics_values.shape[0]

        flatten_mdp_id = self.mdp_id.reshape(-1)
        unique_mdp_ids = set(flatten_mdp_id.tolist())
        prev_mdp_id, prev_seq_num = None, None
        mdp_count = 0
        for mdp_id, seq_num in zip(flatten_mdp_id, self.sequence_number):
            if prev_mdp_id is None or mdp_id != prev_mdp_id:
                mdp_count += 1
                prev_mdp_id = mdp_id
            else:
                assert (
                    seq_num > prev_seq_num
                ), "Sequence number must be in increasing order"

            prev_seq_num = seq_num

        assert len(unique_mdp_ids) == mdp_count, "MDPs are broken up. {} vs {}".format(
            len(unique_mdp_ids), mdp_count
        )

    def set_metric_as_reward(self, i: int, num_actions: int):
        assert self.logged_metrics is not None, "metrics must not be none"
        assert self.logged_metrics_values is not None, "metrics must not be none"
        assert self.model_metrics is not None, "metrics must not be none"
        assert self.model_metrics_values is not None, "metrics must not be none"

        return self._replace(
            logged_rewards=self.logged_metrics[:, i : i + 1],
            logged_values=self.logged_metrics_values[:, i : i + 1],
            model_rewards=self.model_metrics[
                :, i * num_actions : (i + 1) * num_actions
            ],
            model_values=self.model_metrics_values[
                :, i * num_actions : (i + 1) * num_actions
            ],
            logged_metrics=None,
            logged_metrics_values=None,
            model_metrics=None,
            model_metrics_values=None,
        )
