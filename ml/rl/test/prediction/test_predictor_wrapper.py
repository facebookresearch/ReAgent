#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import ml.rl.types as rlt
import torch
from ml.rl.models.actor import FullyConnectedActor
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from ml.rl.prediction.predictor_wrapper import (
    ActorPredictorWrapper,
    ActorWithPreprocessor,
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
    Seq2SlatePredictorWrapper,
    Seq2SlateWithPreprocessor,
)
from ml.rl.preprocessing.identify_types import CONTINUOUS, CONTINUOUS_ACTION
from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.preprocessing.postprocessor import Postprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor


def _cont_norm():
    return NormalizationParameters(feature_type=CONTINUOUS, mean=0.0, stddev=1.0)


def _cont_action_norm():
    return NormalizationParameters(
        feature_type=CONTINUOUS_ACTION, min_value=-3.0, max_value=3.0
    )


class TestPredictorWrapper(unittest.TestCase):
    def test_discrete_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_dim = 2
        dqn = FullyConnectedDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=action_dim,
            sizes=[16],
            activations=["relu"],
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(dqn, state_preprocessor)
        action_names = ["L", "R"]
        wrapper = DiscreteDqnPredictorWrapper(dqn_with_preprocessor, action_names)
        input_prototype = dqn_with_preprocessor.input_prototype()
        output_action_names, q_values = wrapper(*input_prototype)
        self.assertEqual(action_names, output_action_names)
        self.assertEqual(q_values.shape, (1, 2))

        expected_output = dqn(
            rlt.PreprocessedState.from_tensor(state_preprocessor(*input_prototype[0]))
        ).q_values
        self.assertTrue((expected_output == q_values).all())

    def test_parametric_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        action_normalization_parameters = {i: _cont_norm() for i in range(5, 9)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_preprocessor = Preprocessor(action_normalization_parameters, False)
        dqn = FullyConnectedParametricDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=len(action_normalization_parameters),
            sizes=[16],
            activations=["relu"],
        )
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            dqn,
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
        )
        wrapper = ParametricDqnPredictorWrapper(dqn_with_preprocessor)

        input_prototype = dqn_with_preprocessor.input_prototype()
        output_action_names, q_value = wrapper(*input_prototype)
        self.assertEqual(output_action_names, ["Q"])
        self.assertEqual(q_value.shape, (1, 1))

        expected_output = dqn(
            rlt.PreprocessedStateAction.from_tensors(
                state=state_preprocessor(*input_prototype[0]),
                action=action_preprocessor(*input_prototype[1]),
            )
        ).q_value
        self.assertTrue((expected_output == q_value).all())

    def test_actor_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        action_normalization_parameters = {
            i: _cont_action_norm() for i in range(101, 105)
        }
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        postprocessor = Postprocessor(action_normalization_parameters, False)

        # Test with FullyConnectedActor to make behavior deterministic
        actor = FullyConnectedActor(
            state_dim=len(state_normalization_parameters),
            action_dim=len(action_normalization_parameters),
            sizes=[16],
            activations=["relu"],
        )
        actor_with_preprocessor = ActorWithPreprocessor(
            actor, state_preprocessor, postprocessor
        )
        wrapper = ActorPredictorWrapper(actor_with_preprocessor)
        input_prototype = actor_with_preprocessor.input_prototype()
        action = wrapper(*input_prototype)
        self.assertEqual(action.shape, (1, len(action_normalization_parameters)))

        expected_output = postprocessor(
            actor(
                rlt.PreprocessedState.from_tensor(
                    state_preprocessor(*input_prototype[0])
                )
            ).action
        )
        self.assertTrue((expected_output == action).all())

    def test_seq2slate_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        candidate_normalization_parameters = {i: _cont_norm() for i in range(101, 106)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        candidate_preprocessor = Preprocessor(candidate_normalization_parameters, False)
        seq2slate = Seq2SlateTransformerNet(
            state_dim=len(state_normalization_parameters),
            candidate_dim=len(candidate_normalization_parameters),
            num_stacked_layers=2,
            num_heads=2,
            dim_model=10,
            dim_feedforward=10,
            max_src_seq_len=10,
            max_tgt_seq_len=4,
        )
        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate, state_preprocessor, candidate_preprocessor, greedy=True
        )
        wrapper = Seq2SlatePredictorWrapper(seq2slate_with_preprocessor)

        state_input_prototype, candidate_input_prototype = (
            seq2slate_with_preprocessor.input_prototype()
        )
        ret_val = wrapper(state_input_prototype, candidate_input_prototype)

        preprocessed_state = state_preprocessor(
            state_input_prototype[0], state_input_prototype[1]
        )
        preprocessed_candidates = candidate_preprocessor(
            candidate_input_prototype[0].view(
                1 * seq2slate.max_src_seq_len, len(candidate_normalization_parameters)
            ),
            candidate_input_prototype[1].view(
                1 * seq2slate.max_src_seq_len, len(candidate_normalization_parameters)
            ),
        ).view(1, seq2slate.max_src_seq_len, -1)
        src_src_mask = torch.ones(
            1, seq2slate.max_src_seq_len, seq2slate.max_src_seq_len
        )
        ranking_input = rlt.PreprocessedRankingInput.from_tensors(
            state=preprocessed_state,
            src_seq=preprocessed_candidates,
            src_src_mask=src_src_mask,
        )
        expected_output = seq2slate(
            ranking_input,
            mode=Seq2SlateMode.RANK_MODE,
            tgt_seq_len=seq2slate.max_tgt_seq_len,
            greedy=True,
        )
        ranked_tgt_out_probs, ranked_tgt_out_idx = (
            expected_output.ranked_tgt_out_probs,
            expected_output.ranked_tgt_out_idx,
        )
        ranked_tgt_out_probs = torch.prod(
            torch.gather(
                ranked_tgt_out_probs, 2, ranked_tgt_out_idx.unsqueeze(-1)
            ).squeeze(),
            -1,
        )
        # -2 to offset padding symbol and decoder start symbol
        ranked_tgt_out_idx -= 2

        self.assertTrue(ranked_tgt_out_probs == ret_val[0])
        self.assertTrue(torch.all(torch.eq(ret_val[1], ranked_tgt_out_idx)))
