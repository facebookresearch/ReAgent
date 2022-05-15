#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import random
import unittest

import numpy as np
import torch
from reagent.core.parameters import NormalizationData, NormalizationParameters
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateTransformerModel, Seq2SlateTransformerNet
from reagent.prediction.predictor_wrapper import Seq2SlateWithPreprocessor
from reagent.preprocessing.identify_types import DO_NOT_PREPROCESS
from reagent.preprocessing.preprocessor import Preprocessor


logger = logging.getLogger(__name__)


class TestSeq2SlateInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def test_seq2slate_scriptable(self):
        state_dim = 2
        candidate_dim = 3
        num_stacked_layers = 2
        num_heads = 2
        dim_model = 128
        dim_feedforward = 128
        candidate_size = 8
        slate_size = 8
        output_arch = Seq2SlateOutputArch.AUTOREGRESSIVE
        temperature = 1.0
        greedy_serving = True

        # test the raw Seq2Slate model is script-able
        seq2slate = Seq2SlateTransformerModel(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=num_stacked_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            max_src_seq_len=candidate_size,
            max_tgt_seq_len=slate_size,
            output_arch=output_arch,
            temperature=temperature,
        )
        seq2slate_scripted = torch.jit.script(seq2slate)

        seq2slate_net = Seq2SlateTransformerNet(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=num_stacked_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            max_src_seq_len=candidate_size,
            max_tgt_seq_len=slate_size,
            output_arch=output_arch,
            temperature=temperature,
        )

        state_normalization_data = NormalizationData(
            dense_normalization_parameters={
                0: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
                1: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
            }
        )

        candidate_normalization_data = NormalizationData(
            dense_normalization_parameters={
                5: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
                6: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
                7: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
            }
        )
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        candidate_preprocessor = Preprocessor(
            candidate_normalization_data.dense_normalization_parameters, False
        )

        # test seq2slate with preprocessor is scriptable
        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate_net.eval(),
            state_preprocessor,
            candidate_preprocessor,
            greedy_serving,
        )
        torch.jit.script(seq2slate_with_preprocessor)
