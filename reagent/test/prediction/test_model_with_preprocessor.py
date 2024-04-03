#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import numpy.testing as npt
import torch
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.prediction.predictor_wrapper import Seq2SlateWithPreprocessor
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.test.prediction.test_prediction_utils import (
    _cont_norm,
    change_cand_size_slate_ranking,
)


class TestModelWithPreprocessor(unittest.TestCase):
    def verify_results(self, expected_output, scripted_output) -> None:
        for i, j in zip(expected_output, scripted_output):
            npt.assert_array_equal(i.detach(), j.detach())

    def test_seq2slate_transformer_frechet_sort_model_with_preprocessor(self) -> None:
        self._test_seq2slate_model_with_preprocessor(
            model="transformer", output_arch=Seq2SlateOutputArch.FRECHET_SORT
        )

    def test_seq2slate_transformer_autoregressive_model_with_preprocessor(self) -> None:
        self._test_seq2slate_model_with_preprocessor(
            model="transformer", output_arch=Seq2SlateOutputArch.AUTOREGRESSIVE
        )

    def _test_seq2slate_model_with_preprocessor(
        self, model: str, output_arch: Seq2SlateOutputArch
    ) -> None:
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        candidate_normalization_parameters = {i: _cont_norm() for i in range(101, 106)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        candidate_preprocessor = Preprocessor(candidate_normalization_parameters, False)
        candidate_size = 10
        slate_size = 4

        seq2slate = None
        if model == "transformer":
            seq2slate = Seq2SlateTransformerNet(
                state_dim=len(state_normalization_parameters),
                candidate_dim=len(candidate_normalization_parameters),
                num_stacked_layers=2,
                num_heads=2,
                dim_model=10,
                dim_feedforward=10,
                max_src_seq_len=candidate_size,
                max_tgt_seq_len=slate_size,
                output_arch=output_arch,
                temperature=0.5,
            )
        else:
            raise NotImplementedError(f"model type {model} is unknown")

        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate, state_preprocessor, candidate_preprocessor, greedy=True
        )
        input_prototype = seq2slate_with_preprocessor.input_prototype()

        if seq2slate_with_preprocessor.can_be_traced():
            seq2slate_with_preprocessor_jit = torch.jit.trace(
                seq2slate_with_preprocessor,
                seq2slate_with_preprocessor.input_prototype(),
            )
        else:
            seq2slate_with_preprocessor_jit = torch.jit.script(
                seq2slate_with_preprocessor
            )
        expected_output = seq2slate_with_preprocessor(*input_prototype)
        jit_output = seq2slate_with_preprocessor_jit(*input_prototype)
        self.verify_results(expected_output, jit_output)

        # Test if scripted model can handle variable lengths of input
        input_prototype = change_cand_size_slate_ranking(input_prototype, 20)
        expected_output = seq2slate_with_preprocessor(*input_prototype)
        jit_output = seq2slate_with_preprocessor_jit(*input_prototype)
        self.verify_results(expected_output, jit_output)
