#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core import parameters as rlp
from reagent.core import types as rlt
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData, NormalizationParameters
from reagent.net_builder.synthetic_reward.ngram_synthetic_reward import (
    NGramSyntheticReward,
)
from reagent.net_builder.synthetic_reward.single_step_synthetic_reward import (
    SingleStepSyntheticReward,
)
from reagent.net_builder.unions import SyntheticRewardNetBuilder__Union
from reagent.preprocessing.identify_types import CONTINUOUS


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.synthetic_reward.single_step_synthetic_reward import (
        FbParametricSingleStepSyntheticRewardPredictorWrapper as ParametricSingleStepSyntheticRewardPredictorWrapper,
    )
else:
    from reagent.prediction.synthetic_reward.single_step_synthetic_reward import (
        ParametricSingleStepSyntheticRewardPredictorWrapper,
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


class TestSyntheticRewardNetBuilder(unittest.TestCase):
    def test_single_step_synthetic_reward_net_builder_discrete_actions(
        self,
    ):
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

    def test_single_step_synthetic_reward_net_builder_continuous_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            SingleStepSyntheticReward=SingleStepSyntheticReward()
        ).value
        state_normalization_data = _create_norm(STATE_DIM)
        action_normalization_data = _create_norm(ACTION_DIM, offset=STATE_DIM)
        reward_net = builder.build_synthetic_reward_network(
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        input = _create_input()
        output = reward_net(input).predicted_reward
        assert output.shape == (BATCH_SIZE, 1)

        predictor_wrapper = builder.build_serving_module(
            reward_net,
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        self.assertIsInstance(
            predictor_wrapper, ParametricSingleStepSyntheticRewardPredictorWrapper
        )

    def test_ngram_fc_synthetic_reward_net_builder_continuous_actions(
        self,
    ):
        builder = SyntheticRewardNetBuilder__Union(
            NGramSyntheticReward=NGramSyntheticReward()
        ).value
        state_normalization_data = _create_norm(STATE_DIM)
        action_normalization_data = _create_norm(ACTION_DIM, offset=STATE_DIM)
        reward_net = builder.build_synthetic_reward_network(
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        input = _create_input()
        output = reward_net(input).predicted_reward
        assert output.shape == (BATCH_SIZE, 1)

        # TO IMPLEMENT
        # predictor_wrapper = builder.build_serving_module(
        #     reward_net,
        #     state_normalization_data,
        #     action_normalization_data=action_normalization_data,
        # )
        # self.assertIsInstance(
        #     predictor_wrapper, ParametricSingleStepSyntheticRewardPredictorWrapper
        # )

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
            NGramSyntheticReward=NGramSyntheticReward(conv_net_params=conv_net_params)
        ).value
        state_normalization_data = _create_norm(STATE_DIM)
        action_normalization_data = _create_norm(ACTION_DIM, offset=STATE_DIM)
        reward_net = builder.build_synthetic_reward_network(
            state_normalization_data,
            action_normalization_data=action_normalization_data,
        )
        input = _create_input()
        output = reward_net(input).predicted_reward
        assert output.shape == (BATCH_SIZE, 1)

        # TO IMPLEMENT
        # predictor_wrapper = builder.build_serving_module(
        #     reward_net,
        #     state_normalization_data,
        #     action_normalization_data=action_normalization_data,
        # )
        # self.assertIsInstance(
        #     predictor_wrapper, ParametricSingleStepSyntheticRewardPredictorWrapper
        # )
