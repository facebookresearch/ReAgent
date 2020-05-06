#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn.functional as F
from reagent import types as rlt
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor


class InputColumn(object):
    STATE_FEATURES = "state_features"
    STATE_ID_LIST_FEATURES = "state_id_list_features"
    STATE_ID_SCORE_LIST_FEATURES = "state_id_score_list_features"
    NEXT_STATE_FEATURES = "next_state_features"
    NEXT_STATE_ID_LIST_FEATURES = "next_state_id_list_features"
    NEXT_STATE_ID_SCORE_LIST_FEATURES = "next_state_id_score_list_features"
    ACTION = "action"
    NEXT_ACTION = "next_action"
    POSSIBLE_ACTIONS = "possible_actions"
    POSSIBLE_ACTIONS_MASK = "possible_actions_mask"
    POSSIBLE_NEXT_ACTIONS = "possible_next_actions"
    POSSIBLE_NEXT_ACTIONS_MASK = "possible_next_actions_mask"
    NOT_TERMINAL = "not_terminal"
    STEP = "step"
    TIME_DIFF = "time_diff"
    TIME_SINCE_FIRST = "time_since_first"
    MDP_ID = "mdp_id"
    SEQUENCE_NUMBER = "sequence_number"
    METRICS = "metrics"
    REWARD = "reward"
    ACTION_PROBABILITY = "action_probability"
    SLATE_REWARD = "slate_reward"
    POSITION_REWARD = "position_reward"
    CANDIDATE_FEATURES = "candidate_features"
    REWARD_MASK = "reward_mask"
    ITEM_MASK = "item_mask"
    NEXT_ITEM_MASK = "next_item_mask"
    ITEM_PROBABILITY = "item_probability"
    NEXT_ITEM_PROBABILITY = "next_item_probability"
    EXTRAS = "extras"


class BatchPreprocessor:
    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.TensorDataClass:
        raise NotImplementedError()


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k in batch:
        out[k] = batch[k].to(device)
    return out


class DiscreteDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self, num_actions: int, state_preprocessor: Preprocessor, use_gpu: bool
    ):
        self.num_actions = num_actions
        self.state_preprocessor = state_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # TODO: remove type ignore after converting rest of BatchPreprocessors to Dict input
    def __call__(self, batch: Dict[str, torch.Tensor]) -> rlt.DiscreteDqnInput:
        batch = batch_to_device(batch, self.device)
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )
        preprocessed_next_state = self.state_preprocessor(
            batch["next_state_features"], batch["next_state_features_presence"]
        )
        # not terminal iff at least one possible for next action
        not_terminal = batch["possible_next_actions_mask"].max(dim=1)[0].float()
        action = F.one_hot(batch["action"].to(torch.int64), self.num_actions)
        # next action can potentially have value self.num_action if not available
        next_action = F.one_hot(
            batch["next_action"].to(torch.int64), self.num_actions + 1
        )[:, : self.num_actions]
        return rlt.DiscreteDqnInput(
            state=rlt.PreprocessedFeatureVector(preprocessed_state),
            next_state=rlt.PreprocessedFeatureVector(preprocessed_next_state),
            action=action,
            next_action=next_action,
            reward=batch["reward"].unsqueeze(1),
            time_diff=batch["time_diff"].unsqueeze(1),
            step=batch["step"].unsqueeze(1),
            not_terminal=not_terminal.unsqueeze(1),
            possible_actions_mask=batch["possible_actions_mask"],
            possible_next_actions_mask=batch["possible_next_actions_mask"],
            extras=rlt.ExtraData(
                mdp_id=batch["mdp_id"].unsqueeze(1),
                sequence_number=batch["sequence_number"].unsqueeze(1),
                action_probability=batch["action_probability"].unsqueeze(1),
            ),
        )


class SequentialDiscreteDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(self, state_preprocessor: Preprocessor, action_dim: int, seq_len: int):
        self.state_preprocessor = state_preprocessor
        self.state_dim = get_num_output_features(
            state_preprocessor.normalization_parameters
        )
        self.seq_len = seq_len
        self.action_dim = action_dim

    def __call__(self, batch: rlt.RawTrainingBatch) -> rlt.PreprocessedTrainingBatch:
        training_input = batch.training_input
        assert isinstance(
            training_input, rlt.RawMemoryNetworkInput
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
        preprocessed_batch = batch.preprocess(new_training_input)
        assert isinstance(new_training_input, rlt.PreprocessedMemoryNetworkInput)
        preprocessed_batch = preprocessed_batch._replace(
            training_input=new_training_input._replace(
                state=rlt.PreprocessedFeatureVector(
                    float_features=new_training_input.state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                action=new_training_input.action.reshape(
                    -1, self.seq_len, self.action_dim
                ),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=new_training_input.next_state.float_features.reshape(
                        -1, self.seq_len, self.state_dim
                    )
                ),
                reward=new_training_input.reward.reshape(-1, self.seq_len),
                not_terminal=new_training_input.not_terminal.reshape(-1, self.seq_len),
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
        self,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        use_gpu: bool,
    ):
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # TODO: remove type ignore after converting rest of BatchPreprocessors to Dict input
    def __call__(self, batch: Dict[str, torch.Tensor]) -> rlt.PolicyNetworkInput:
        batch = batch_to_device(batch, self.device)
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )
        preprocessed_next_state = self.state_preprocessor(
            batch["next_state_features"], batch["next_state_features_presence"]
        )
        preprocessed_action = self.action_preprocessor(
            batch["action"], batch["action_presence"]
        )
        preprocessed_next_action = self.action_preprocessor(
            batch["next_action"], batch["next_action_presence"]
        )
        return rlt.PolicyNetworkInput(
            state=rlt.PreprocessedFeatureVector(preprocessed_state),
            next_state=rlt.PreprocessedFeatureVector(preprocessed_next_state),
            action=rlt.PreprocessedFeatureVector(preprocessed_action),
            next_action=rlt.PreprocessedFeatureVector(preprocessed_next_action),
            reward=batch["reward"].unsqueeze(1),
            time_diff=batch["time_diff"].unsqueeze(1),
            step=batch["step"].unsqueeze(1),
            not_terminal=batch["not_terminal"].unsqueeze(1),
            extras=rlt.ExtraData(
                mdp_id=batch["mdp_id"].unsqueeze(1),
                sequence_number=batch["sequence_number"].unsqueeze(1),
                action_probability=batch["action_probability"].unsqueeze(1),
            ),
        )
