#!/usr/bin/env python3

from typing import Tuple, Union, cast

import torch
from ml.rl import types as rlt
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.preprocessing.preprocessor import Preprocessor


class BatchPreprocessor:
    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        raise NotImplementedError()


class DiscreteDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(self, state_preprocessor: Preprocessor):
        self.state_preprocessor = state_preprocessor

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        training_input = batch.training_input
        assert isinstance(
            training_input, (rlt.RawDiscreteDqnInput, rlt.RawMemoryNetworkInput)
        ), "Wrong Type: {}".format(str(type(training_input)))

        preprocessed_state = self.state_preprocessor(
            training_input.state.float_features.value,
            training_input.state.float_features.presence,
        )
        preprocessed_next_state = self.state_preprocessor(
            training_input.next_state.float_features.value,
            training_input.next_state.float_features.presence,
        )
        new_training_input = training_input.preprocess_tensors(
            state=preprocessed_state, next_state=preprocessed_next_state
        )
        return batch.preprocess(new_training_input)


class SequentialDiscreteDqnBatchPreprocessor(DiscreteDqnBatchPreprocessor):
    def __init__(self, state_preprocessor: Preprocessor, action_dim: int, seq_len: int):
        super().__init__(state_preprocessor)
        self.state_dim = get_num_output_features(
            state_preprocessor.normalization_parameters
        )
        self.seq_len = seq_len
        self.action_dim = action_dim

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        preprocessed_batch = super().__call__(batch)
        training_input = preprocessed_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedMemoryNetworkInput)
        preprocessed_batch = preprocessed_batch._replace(
            training_input=training_input._replace(
                state=rlt.PreprocessedFeatureVector(
                    float_features=training_input.state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                action=training_input.action.reshape(-1, self.seq_len, self.action_dim),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=training_input.next_state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                reward=training_input.reward.reshape(-1, self.seq_len),
                not_terminal=training_input.not_terminal.reshape(-1, self.seq_len),
            )
        )
        return preprocessed_batch


class ParametricDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self, state_preprocessor: Preprocessor, action_preprocessor: Preprocessor
    ):
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        training_input = batch.training_input
        assert isinstance(
            training_input, (rlt.RawParametricDqnInput, rlt.RawMemoryNetworkInput)
        ), "Wrong Type: {}".format(str(type(training_input)))
        is_memory_network = isinstance(training_input, rlt.RawMemoryNetworkInput)
        preprocessed_state = self.state_preprocessor(
            training_input.state.float_features.value,
            training_input.state.float_features.presence,
        )
        preprocessed_next_state = self.state_preprocessor(
            training_input.next_state.float_features.value,
            training_input.next_state.float_features.presence,
        )
        assert isinstance(training_input.action, rlt.FeatureVector)
        preprocessed_action = self.action_preprocessor(
            training_input.action.float_features.value,
            training_input.action.float_features.presence,
        )
        if is_memory_network:
            assert isinstance(training_input, rlt.RawMemoryNetworkInput)
            return batch.preprocess(
                training_input=training_input.preprocess_tensors(
                    state=preprocessed_state,
                    next_state=preprocessed_next_state,
                    action=preprocessed_action,
                )
            )
        else:
            assert isinstance(training_input, rlt.RawParametricDqnInput)
            preprocessed_tiled_next_state = self.state_preprocessor(
                training_input.tiled_next_state.float_features.value,
                training_input.tiled_next_state.float_features.presence,
            )
            preprocessed_next_action = self.action_preprocessor(
                training_input.next_action.float_features.value,
                training_input.next_action.float_features.presence,
            )
            preprocessed_possible_actions = self.action_preprocessor(
                training_input.possible_actions.float_features.value,
                training_input.possible_actions.float_features.presence,
            )
            preprocessed_possible_next_actions = self.action_preprocessor(
                training_input.possible_next_actions.float_features.value,
                training_input.possible_next_actions.float_features.presence,
            )
            return batch.preprocess(
                training_input=training_input.preprocess_tensors(
                    state=preprocessed_state,
                    next_state=preprocessed_next_state,
                    action=preprocessed_action,
                    next_action=preprocessed_next_action,
                    possible_actions=preprocessed_possible_actions,
                    possible_next_actions=preprocessed_possible_next_actions,
                    tiled_next_state=preprocessed_tiled_next_state,
                )
            )


class SequentialParametricDqnBatchPreprocessor(ParametricDqnBatchPreprocessor):
    def __init__(
        self,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        seq_len: int,
    ):
        super().__init__(state_preprocessor, action_preprocessor)
        self.state_dim = get_num_output_features(
            state_preprocessor.normalization_parameters
        )
        self.action_dim = get_num_output_features(
            action_preprocessor.normalization_parameters
        )
        self.seq_len = seq_len

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        preprocessed_batch = super().__call__(batch)
        training_input = preprocessed_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedMemoryNetworkInput)
        preprocessed_batch = preprocessed_batch._replace(
            training_input=training_input._replace(
                state=rlt.PreprocessedFeatureVector(
                    float_features=training_input.state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                action=training_input.action.reshape(-1, self.seq_len, self.action_dim),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=training_input.next_state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                reward=training_input.reward.reshape(-1, self.seq_len),
                not_terminal=training_input.not_terminal.reshape(-1, self.seq_len),
            )
        )
        return preprocessed_batch


class PolicyNetworkBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self, state_preprocessor: Preprocessor, action_preprocessor: Preprocessor
    ):
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        training_input = batch.training_input
        assert isinstance(training_input, rlt.RawPolicyNetworkInput)

        preprocessed_state = self.state_preprocessor(
            training_input.state.float_features.value,
            training_input.state.float_features.presence,
        )
        preprocessed_next_state = self.state_preprocessor(
            training_input.next_state.float_features.value,
            training_input.next_state.float_features.presence,
        )
        preprocessed_action = self.action_preprocessor(
            training_input.action.float_features.value,
            training_input.action.float_features.presence,
        )
        preprocessed_next_action = self.action_preprocessor(
            training_input.next_action.float_features.value,
            training_input.next_action.float_features.presence,
        )
        return batch.preprocess(
            training_input=training_input.preprocess_tensors(
                state=preprocessed_state,
                next_state=preprocessed_next_state,
                action=preprocessed_action,
                next_action=preprocessed_next_action,
            )
        )
