#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest

import numpy.testing as npt
import reagent.core.types as rlt
import reagent.models as models
import torch
from reagent.model_utils.seq2slate_utils import Seq2SlateMode, Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.prediction.cfeval.predictor_wrapper import (
    BanditRewardNetPredictorWrapper,
)
from reagent.prediction.predictor_wrapper import (
    ActorPredictorWrapper,
    ActorWithPreprocessor,
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
    Seq2SlatePredictorWrapper,
    Seq2SlateWithPreprocessor,
)
from reagent.prediction.ranking.predictor_wrapper import (
    DeterminantalPointProcessPredictorWrapper,
    Kernel,
)
from reagent.preprocessing.postprocessor import Postprocessor
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.test.prediction.test_prediction_utils import _cont_norm, _cont_action_norm
from reagent.test.prediction.test_prediction_utils import (
    change_cand_size_slate_ranking,
)


def seq2slate_input_prototype_to_ranking_input(
    state_input_prototype,
    candidate_input_prototype,
    state_preprocessor,
    candidate_preprocessor,
):
    batch_size, candidate_size, candidate_dim = candidate_input_prototype[0].shape
    preprocessed_state = state_preprocessor(
        state_input_prototype[0], state_input_prototype[1]
    )
    preprocessed_candidates = candidate_preprocessor(
        candidate_input_prototype[0].view(batch_size * candidate_size, candidate_dim),
        candidate_input_prototype[1].view(batch_size * candidate_size, candidate_dim),
    ).view(batch_size, candidate_size, -1)
    return rlt.PreprocessedRankingInput.from_tensors(
        state=preprocessed_state,
        src_seq=preprocessed_candidates,
    )


class TestPredictorWrapper(unittest.TestCase):
    def test_discrete_wrapper(self):
        ids = range(1, 5)
        state_normalization_parameters = {i: _cont_norm() for i in ids}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_dim = 2
        dqn = models.FullyConnectedDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=action_dim,
            sizes=[16],
            activations=["relu"],
        )
        state_feature_config = rlt.ModelFeatureConfig(
            float_feature_infos=[
                rlt.FloatFeatureInfo(feature_id=i, name=f"feat_{i}") for i in ids
            ]
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            dqn, state_preprocessor, state_feature_config
        )
        action_names = ["L", "R"]
        wrapper = DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names, state_feature_config
        )
        input_prototype = dqn_with_preprocessor.input_prototype()[0]
        output_action_names, q_values = wrapper(input_prototype)
        self.assertEqual(action_names, output_action_names)
        self.assertEqual(q_values.shape, (1, 2))

        state_with_presence = input_prototype.float_features_with_presence
        expected_output = dqn(rlt.FeatureData(state_preprocessor(*state_with_presence)))
        self.assertTrue((expected_output == q_values).all())

    def test_discrete_wrapper_with_id_list(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_dim = 2
        state_feature_config = rlt.ModelFeatureConfig(
            float_feature_infos=[
                rlt.FloatFeatureInfo(name=str(i), feature_id=i) for i in range(1, 5)
            ],
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name="A", feature_id=10, id_mapping_name="A_mapping"
                )
            ],
            id_mapping_config={
                "A_mapping": rlt.IdMappingUnion(
                    explicit_mapping=rlt.ExplicitMapping(ids=[0, 1, 2])
                )
            },
        )
        embedding_concat = models.EmbeddingBagConcat(
            state_dim=len(state_normalization_parameters),
            model_feature_config=state_feature_config,
            embedding_dim=8,
        )
        dqn = models.Sequential(
            embedding_concat,
            rlt.TensorFeatureData(),
            models.FullyConnectedDQN(
                embedding_concat.output_dim,
                action_dim=action_dim,
                sizes=[16],
                activations=["relu"],
            ),
        )

        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            dqn, state_preprocessor, state_feature_config
        )
        action_names = ["L", "R"]
        wrapper = DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names, state_feature_config
        )
        input_prototype = dqn_with_preprocessor.input_prototype()[0]
        output_action_names, q_values = wrapper(input_prototype)
        self.assertEqual(action_names, output_action_names)
        self.assertEqual(q_values.shape, (1, 2))

        feature_id_to_name = {
            config.feature_id: config.name
            for config in state_feature_config.id_list_feature_configs
        }
        state_id_list_features = {
            feature_id_to_name[k]: v
            for k, v in input_prototype.id_list_features.items()
        }
        state_with_presence = input_prototype.float_features_with_presence
        expected_output = dqn(
            rlt.FeatureData(
                float_features=state_preprocessor(*state_with_presence),
                id_list_features=state_id_list_features,
            )
        )
        self.assertTrue((expected_output == q_values).all())

    def test_parametric_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        action_normalization_parameters = {i: _cont_norm() for i in range(5, 9)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_preprocessor = Preprocessor(action_normalization_parameters, False)
        dqn = models.FullyConnectedCritic(
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
            rlt.FeatureData(state_preprocessor(*input_prototype[0])),
            rlt.FeatureData(action_preprocessor(*input_prototype[1])),
        )
        self.assertTrue((expected_output == q_value).all())

    def test_actor_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        action_normalization_parameters = {
            i: _cont_action_norm() for i in range(101, 105)
        }
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        postprocessor = Postprocessor(action_normalization_parameters, False)

        # Test with FullyConnectedActor to make behavior deterministic
        actor = models.FullyConnectedActor(
            state_dim=len(state_normalization_parameters),
            action_dim=len(action_normalization_parameters),
            sizes=[16],
            activations=["relu"],
        )
        state_feature_config = rlt.ModelFeatureConfig()
        actor_with_preprocessor = ActorWithPreprocessor(
            actor, state_preprocessor, state_feature_config, postprocessor
        )
        wrapper = ActorPredictorWrapper(actor_with_preprocessor, state_feature_config)
        input_prototype = actor_with_preprocessor.input_prototype()[0]
        action, _log_prob = wrapper(input_prototype)
        self.assertEqual(action.shape, (1, len(action_normalization_parameters)))

        expected_output = postprocessor(
            actor(rlt.FeatureData(state_preprocessor(*input_prototype[0]))).action
        )
        self.assertTrue((expected_output == action).all())

    def validate_seq2slate_output(self, expected_output, wrapper_output):
        ranked_per_seq_probs, ranked_tgt_out_idx = (
            expected_output.ranked_per_seq_probs,
            expected_output.ranked_tgt_out_idx,
        )
        # -2 to offset padding symbol and decoder start symbol
        ranked_tgt_out_idx -= 2

        self.assertTrue(ranked_per_seq_probs == wrapper_output[0])
        self.assertTrue(torch.all(torch.eq(ranked_tgt_out_idx, wrapper_output[1])))

    def test_seq2slate_transformer_frechet_sort_wrapper(self):
        self._test_seq2slate_wrapper(
            model="transformer", output_arch=Seq2SlateOutputArch.FRECHET_SORT
        )

    def test_seq2slate_transformer_autoregressive_wrapper(self):
        self._test_seq2slate_wrapper(
            model="transformer", output_arch=Seq2SlateOutputArch.AUTOREGRESSIVE
        )

    def _test_seq2slate_wrapper(self, model: str, output_arch: Seq2SlateOutputArch):
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
        wrapper = Seq2SlatePredictorWrapper(seq2slate_with_preprocessor)

        (
            state_input_prototype,
            candidate_input_prototype,
        ) = seq2slate_with_preprocessor.input_prototype()
        wrapper_output = wrapper(state_input_prototype, candidate_input_prototype)

        ranking_input = seq2slate_input_prototype_to_ranking_input(
            state_input_prototype,
            candidate_input_prototype,
            state_preprocessor,
            candidate_preprocessor,
        )
        expected_output = seq2slate(
            ranking_input,
            mode=Seq2SlateMode.RANK_MODE,
            tgt_seq_len=candidate_size,
            greedy=True,
        )
        self.validate_seq2slate_output(expected_output, wrapper_output)

        # Test Seq2SlatePredictorWrapper can handle variable lengths of inputs
        random_length = random.randint(candidate_size + 1, candidate_size * 2)
        (
            state_input_prototype,
            candidate_input_prototype,
        ) = change_cand_size_slate_ranking(
            seq2slate_with_preprocessor.input_prototype(), random_length
        )
        wrapper_output = wrapper(state_input_prototype, candidate_input_prototype)

        ranking_input = seq2slate_input_prototype_to_ranking_input(
            state_input_prototype,
            candidate_input_prototype,
            state_preprocessor,
            candidate_preprocessor,
        )
        expected_output = seq2slate(
            ranking_input,
            mode=Seq2SlateMode.RANK_MODE,
            tgt_seq_len=random_length,
            greedy=True,
        )
        self.validate_seq2slate_output(expected_output, wrapper_output)

    def test_determinantal_point_process_wrapper_linear_kernel(self):
        # The second and third items are identical (similarity=1)
        # So the second and third items have strong repulsion
        # The expected ranked indices should be 2, 0, 1
        quality_scores = torch.tensor(
            [
                [4],
                [5],
                [8],
            ]
        )

        feature_vectors = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])

        wrapper = DeterminantalPointProcessPredictorWrapper(
            alpha=1.0, kernel=Kernel.Linear
        )
        ranked_idx, determinants, L = wrapper(quality_scores, feature_vectors)
        npt.assert_array_almost_equal(ranked_idx, [2, 0, 1])
        npt.assert_array_almost_equal(
            determinants,
            torch.tensor(
                [
                    [16, 25, 64],
                    [1024, 0, wrapper.MIN_VALUE],
                    [wrapper.MIN_VALUE, 0, wrapper.MIN_VALUE],
                ]
            ),
        )
        npt.assert_array_almost_equal(L, [[16, 0, 0], [0, 25, 40], [0, 40, 64]])

        # Test shorter rerank positions
        # All three items have different categories, so the final order is 1, 2, 0 if
        # rerank the full slate. If rerank_topk=1, then the expected order is 1, 0, 2
        quality_scores = torch.tensor(
            [
                [4],
                [6],
                [5],
            ]
        )
        feature_vectors = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        wrapper = DeterminantalPointProcessPredictorWrapper(
            alpha=1.0, kernel=Kernel.Linear, rerank_topk=1
        )
        ranked_idx, _, _ = wrapper(quality_scores, feature_vectors)
        npt.assert_array_almost_equal(ranked_idx, [1, 0, 2])

    def test_determinantal_point_process_wrapper_rbf_kernel(self):
        # The second and third items are identical (similarity=1)
        # So the second and third items have strong repulsion
        # The expected ranked indices should be 2, 0, 1
        quality_scores = torch.tensor(
            [
                [4],
                [5],
                [8],
            ]
        )

        feature_vectors = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])

        wrapper = DeterminantalPointProcessPredictorWrapper(
            alpha=1.0, kernel=Kernel.RBF
        )
        ranked_idx, determinants, L = wrapper(quality_scores, feature_vectors)
        npt.assert_array_almost_equal(ranked_idx, [2, 0, 1])
        npt.assert_array_almost_equal(
            determinants,
            torch.tensor(
                [
                    [16, 25, 64],
                    [885.41766159, 0, wrapper.MIN_VALUE],
                    [wrapper.MIN_VALUE, 0, wrapper.MIN_VALUE],
                ]
            ),
            decimal=3,
        )
        npt.assert_array_almost_equal(
            L, [[16, 7.3576, 11.7721], [7.3576, 25, 40], [11.7721, 40, 64]], decimal=3
        )

        # Test shorter rerank positions
        # All three items have different categories, so the final order is 1, 2, 0 if
        # rerank the full slate. If rerank_topk=1, then the expected order is 1, 0, 2
        quality_scores = torch.tensor(
            [
                [4],
                [6],
                [5],
            ]
        )
        feature_vectors = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        wrapper = DeterminantalPointProcessPredictorWrapper(
            alpha=1.0, kernel=Kernel.RBF, rerank_topk=1
        )
        ranked_idx, _, _ = wrapper(quality_scores, feature_vectors)
        npt.assert_array_almost_equal(ranked_idx, [1, 0, 2])

    def test_reward_model_wrapper(self):
        ids = range(1, 5)
        state_normalization_parameters = {i: _cont_norm() for i in ids}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_dim = 2
        model = models.FullyConnectedDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=action_dim,
            sizes=[16],
            activations=["relu"],
        )
        state_feature_config = rlt.ModelFeatureConfig(
            float_feature_infos=[
                rlt.FloatFeatureInfo(feature_id=i, name=f"feat_{i}") for i in ids
            ]
        )
        model_with_preprocessor = DiscreteDqnWithPreprocessor(
            model, state_preprocessor, state_feature_config
        )
        action_names = ["L", "R"]
        wrapper = BanditRewardNetPredictorWrapper(
            model_with_preprocessor, action_names, state_feature_config
        )
        input_prototype = model_with_preprocessor.input_prototype()[0]
        reward_predictions, mask = wrapper(input_prototype)
        self.assertEqual(reward_predictions.shape, (1, 2))

        state_with_presence = input_prototype.float_features_with_presence
        expected_output = model(
            rlt.FeatureData(state_preprocessor(*state_with_presence))
        )
        self.assertTrue((expected_output == reward_predictions).all())
