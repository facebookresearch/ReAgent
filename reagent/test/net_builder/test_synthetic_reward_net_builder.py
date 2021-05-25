#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy.testing as npt
import torch
from reagent.core import parameters as rlp
from reagent.core import types as rlt
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData, NormalizationParameters
from reagent.net_builder.synthetic_reward.ngram_synthetic_reward import (
    NGramSyntheticReward,
    NGramConvNetSyntheticReward,
)
from reagent.net_builder.synthetic_reward.sequence_synthetic_reward import (
    SequenceSyntheticReward,
)
from reagent.net_builder.synthetic_reward.single_step_synthetic_reward import (
    SingleStepSyntheticReward,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.net_builder.unions import SyntheticRewardNetBuilder__Union
from reagent.preprocessing.identify_types import CONTINUOUS
from reagent.preprocessing.preprocessor import Preprocessor


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.synthetic_reward.synthetic_reward_predictor_wrapper import (
        FbSyntheticRewardPredictorWrapper as SyntheticRewardPredictorWrapper,
    )
else:
    from reagent.prediction.synthetic_reward.synthetic_reward_predictor_wrapper import (
        SyntheticRewardPredictorWrapper,
    )

STATE_DIM = 3
ACTION_DIM = 2
BATCH_SIZE = 2
SEQ_LEN = 4


def _create_norm(dim, offset=0):
    normalization_data = NormalizationData(
        dense_normalization_parameters={
            i: NormalizationParameters(feature_type=CONTINUOUS, mean=0.0, stddev=1.0)
            for i in range(offset, dim + offset)
        }
    )
    return normalization_data


def _create_input():
    state = torch.randn(SEQ_LEN, BATCH_SIZE, STATE_DIM)
    valid_step = torch.tensor([[1], [4]])
    action = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    )
    input = rlt.MemoryNetworkInput(
        state=rlt.FeatureData(state),
        action=action,
        valid_step=valid_step,
        # the rest fields will not be used
        next_state=torch.tensor([]),
        reward=torch.tensor([]),
        step=torch.tensor([]),
        not_terminal=torch.tensor([]),
        time_diff=torch.tensor([]),
    )
    return input


def _create_preprocessed_input(
    input: rlt.MemoryNetworkInput,
    state_preprocessor: Preprocessor,
    action_preprocessor: Preprocessor,
):
    preprocessed_state = state_preprocessor(
        input.state.float_features.reshape(SEQ_LEN * BATCH_SIZE, STATE_DIM),
        torch.ones(SEQ_LEN * BATCH_SIZE, STATE_DIM),
    ).reshape(SEQ_LEN, BATCH_SIZE, STATE_DIM)
    preprocessed_action = action_preprocessor(
        input.action.reshape(SEQ_LEN * BATCH_SIZE, ACTION_DIM),
        torch.ones(SEQ_LEN * BATCH_SIZE, ACTION_DIM),
    ).reshape(SEQ_LEN, BATCH_SIZE, ACTION_DIM)
    return rlt.MemoryNetworkInput(
        state=rlt.FeatureData(preprocessed_state),
        action=preprocessed_action,
        valid_step=input.valid_step,
        next_state=input.next_state,
        reward=input.reward,
        step=input.step,
        not_terminal=input.not_terminal,
        time_diff=input.time_diff,
    )


class TestSyntheticRewardNetBuilder(unittest.TestCase):
    def test_single_step_synthetic_reward_net_builder_discrete_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            SingleStepSyntheticReward=SingleStepSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_discrete_actions(builder)

    def test_ngram_fc_synthetic_reward_net_builder_discrete_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            NGramSyntheticReward=NGramSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_discrete_actions(builder)

    def test_ngram_conv_net_synthetic_reward_net_builder_discrete_actions(
        self,
    ):
        conv_net_params = rlp.ConvNetParameters(
            conv_dims=[256, 128],
            conv_height_kernels=[1, 1],
            pool_types=["max", "max"],
            pool_kernel_sizes=[1, 1],
        )
        builder = SyntheticRewardNetBuilder__Union(
            NGramConvNetSyntheticReward=NGramConvNetSyntheticReward(
                conv_net_params=conv_net_params
            )
        ).value
        self._test_synthetic_reward_net_builder_discrete_actions(builder)

    def test_lstm_synthetic_reward_net_builder_discrete_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            SequenceSyntheticReward=SequenceSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_discrete_actions(builder)

    def _test_synthetic_reward_net_builder_discrete_actions(
        self, builder: SyntheticRewardNetBuilder
    ):
        # pyre-fixme[28]: Unexpected keyword argument `SingleStepSyntheticReward`.
        builder = SyntheticRewardNetBuilder__Union(
            SingleStepSyntheticReward=SingleStepSyntheticReward()
        ).value
        state_normalization_data = _create_norm(STATE_DIM)
        discrete_action_names = ["1", "2"]
        reward_net = builder.build_synthetic_reward_network(
            state_normalization_data, discrete_action_names=discrete_action_names
        )
        input = _create_input()
        output = reward_net(input).predicted_reward
        assert output.shape == (BATCH_SIZE, 1)

        # TO IMPLEMENT
        # predictor_wrapper = builder.build_serving_module(
        #     reward_net,
        #     state_normalization_data,
        #     discrete_action_names=discrete_action_names,
        # )
        # self.assertIsInstance(
        #     predictor_wrapper, DiscreteSingleStepSyntheticRewardPredictorWrapper
        # )

    def test_single_step_synthetic_reward_net_builder_continuous_actions(self):
        builder = SyntheticRewardNetBuilder__Union(
            SingleStepSyntheticReward=SingleStepSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_continuous_actions(builder)

    def test_ngram_fc_synthetic_reward_net_builder_continuous_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            NGramSyntheticReward=NGramSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_continuous_actions(builder)

    def test_ngram_conv_net_synthetic_reward_net_builder_continuous_actions(
        self,
    ):
        conv_net_params = rlp.ConvNetParameters(
            conv_dims=[256, 128],
            conv_height_kernels=[1, 1],
            pool_types=["max", "max"],
            pool_kernel_sizes=[1, 1],
        )
        builder = SyntheticRewardNetBuilder__Union(
            NGramConvNetSyntheticReward=NGramConvNetSyntheticReward(
                conv_net_params=conv_net_params
            )
        ).value
        self._test_synthetic_reward_net_builder_continuous_actions(builder)

    def test_lstm_synthetic_reward_net_builder_continuous_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            SequenceSyntheticReward=SequenceSyntheticReward()
        ).value
        self._test_synthetic_reward_net_builder_continuous_actions(builder)

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def _test_synthetic_reward_net_builder_continuous_actions(
        self, builder: SyntheticRewardNetBuilder
    ):
        """
        This test does the following steps:
        1. create a net builder
        2. use the net builder to create a synthetic reward network
        3. export the synthetic reward network
        4. use the exported network to create a predictor wrapper
        5. create raw input and preprocessed inputs
        6. compare if the results between the following matches:
            a. synthetic reward network on preprocessed input
            b. export network on preprocessed input
            c. predictor wrapper on raw input
        """
        state_normalization_data = _create_norm(STATE_DIM)
        action_normalization_data = _create_norm(ACTION_DIM, offset=STATE_DIM)
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters
        )
        action_preprocessor = Preprocessor(
            action_normalization_data.dense_normalization_parameters
        )
        reward_net = builder.build_synthetic_reward_network(
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        input = _create_input()
        preprocessed_input = _create_preprocessed_input(
            input, state_preprocessor, action_preprocessor
        )
        output = reward_net(preprocessed_input).predicted_reward
        assert output.shape == (BATCH_SIZE, 1)

        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        export_net = reward_net.export_mlp().cpu().eval()
        export_output = export_net(
            preprocessed_input.state.float_features, preprocessed_input.action
        )
        predictor_wrapper = builder.build_serving_module(
            SEQ_LEN,
            reward_net,
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        self.assertIsInstance(predictor_wrapper, SyntheticRewardPredictorWrapper)
        for i in range(BATCH_SIZE):
            input_to_predictor = torch.cat(
                (input.state.float_features[:, i, :], input.action[:, i, :]), dim=1
            )
            input_to_predictor_presence = torch.ones(SEQ_LEN, STATE_DIM + ACTION_DIM)
            predictor_output = predictor_wrapper(
                (input_to_predictor, input_to_predictor_presence)
            )
            if IS_FB_ENVIRONMENT:
                predictor_output = predictor_output[1][2]
            npt.assert_array_almost_equal(predictor_output, export_output[i], decimal=4)
            npt.assert_almost_equal(
                torch.sum(predictor_output[-input.valid_step[i] :]),
                output[i],
                decimal=4,
            )
