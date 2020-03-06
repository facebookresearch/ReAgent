#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
import torch


@dataclass
class BaseDataClass:
    def _replace(self, **kwargs):
        return cast(type(self), dataclasses.replace(self, **kwargs))

    def pin_memory(self):
        pinned_memory = {}
        for field in dataclasses.fields(self):
            f = getattr(self, field.name)
            if isinstance(f, (torch.Tensor, BaseDataClass)):
                pinned_memory[field.name] = f.pin_memory()
        return self._replace(**pinned_memory)

    def cuda(self):
        cuda_tensor = {}
        for field in dataclasses.fields(self):
            f = getattr(self, field.name)
            if isinstance(f, torch.Tensor):
                cuda_tensor[field.name] = f.cuda(non_blocking=True)
            elif isinstance(f, BaseDataClass):
                cuda_tensor[field.name] = f.cuda()
        return self._replace(**cuda_tensor)


@dataclass
class ValuePresence(BaseDataClass):
    value: torch.Tensor
    presence: Optional[torch.Tensor]


@dataclass
class IdListFeatureConfig(BaseDataClass):
    """
    This describes how to map raw features to model features
    """

    name: str
    feature_id: int  # integer feature ID
    id_mapping_name: str  # key to ModelPreprocessingConfig.id_mapping_config
    # max_length: int


@dataclass
class FloatFeatureInfo(BaseDataClass):
    name: str
    feature_id: int


@dataclass
class IdMapping(BaseDataClass):
    ids: List[int]


@dataclass
class ModelFeatureConfig(BaseDataClass):
    float_feature_infos: List[FloatFeatureInfo]
    id_mapping_config: Dict[str, IdMapping] = dataclasses.field(default_factory=dict)
    id_list_feature_configs: List[IdListFeatureConfig] = dataclasses.field(
        default_factory=list
    )


IdListFeatureValue = Tuple[torch.Tensor, torch.Tensor]
IdListFeatures = Dict[str, IdListFeatureValue]


@dataclass
class FeatureVector(BaseDataClass):
    float_features: ValuePresence
    id_list_features: IdListFeatures = dataclasses.field(default_factory=dict)
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[torch.Tensor] = None


@dataclass
class ActorOutput(BaseDataClass):
    action: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    action_mean: Optional[torch.Tensor] = None


@dataclass
class PreprocessedFeatureVector(BaseDataClass):
    float_features: torch.Tensor
    id_list_features: IdListFeatures = dataclasses.field(default_factory=dict)
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[torch.Tensor] = None


@dataclass
class PreprocessedTiledFeatureVector(BaseDataClass):
    float_features: torch.Tensor

    def as_preprocessed_feature_vector(self) -> PreprocessedFeatureVector:
        return PreprocessedFeatureVector(
            float_features=self.float_features.view(-1, self.float_features.shape[2])
        )


@dataclass
class PreprocessedState(BaseDataClass):
    """
    This class makes it easier to plug modules into predictor
    """

    state: PreprocessedFeatureVector

    @classmethod
    def from_tensor(cls, state: torch.Tensor):
        assert isinstance(state, torch.Tensor)
        return cls(state=PreprocessedFeatureVector(float_features=state))

    def __init__(self, state):
        super().__init__()
        if isinstance(state, torch.Tensor):
            raise ValueError("Use from_tensor()")
        self.state = state


@dataclass
class PreprocessedStateAction(BaseDataClass):
    state: PreprocessedFeatureVector
    action: PreprocessedFeatureVector

    @classmethod
    def from_tensors(cls, state: torch.Tensor, action: torch.Tensor):
        assert isinstance(state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        return cls(
            state=PreprocessedFeatureVector(float_features=state),
            action=PreprocessedFeatureVector(float_features=action),
        )

    def __init__(self, state, action):
        super().__init__()
        if isinstance(state, torch.Tensor) or isinstance(action, torch.Tensor):
            raise ValueError(f"Use from_tensors() {type(state)} {type(action)}")
        self.state = state
        self.action = action


@dataclass
class PreprocessedRankingInput(BaseDataClass):
    state: PreprocessedFeatureVector
    src_seq: PreprocessedFeatureVector
    src_src_mask: torch.Tensor
    tgt_in_seq: Optional[PreprocessedFeatureVector] = None
    tgt_out_seq: Optional[PreprocessedFeatureVector] = None
    tgt_tgt_mask: Optional[torch.Tensor] = None
    slate_reward: Optional[torch.Tensor] = None
    position_reward: Optional[torch.Tensor] = None
    # all indices will be +2 to account for padding
    # symbol (0) and decoder_start_symbol (1)
    src_in_idx: Optional[torch.Tensor] = None
    tgt_in_idx: Optional[torch.Tensor] = None
    tgt_out_idx: Optional[torch.Tensor] = None
    tgt_out_probs: Optional[torch.Tensor] = None
    # store ground-truth target sequences
    optim_tgt_in_idx: Optional[torch.Tensor] = None
    optim_tgt_out_idx: Optional[torch.Tensor] = None
    optim_tgt_in_seq: Optional[PreprocessedFeatureVector] = None
    optim_tgt_out_seq: Optional[PreprocessedFeatureVector] = None

    @classmethod
    def from_tensors(
        cls,
        state: torch.Tensor,
        src_seq: torch.Tensor,
        src_src_mask: torch.Tensor,
        tgt_in_seq: Optional[torch.Tensor] = None,
        tgt_out_seq: Optional[torch.Tensor] = None,
        tgt_tgt_mask: Optional[torch.Tensor] = None,
        slate_reward: Optional[torch.Tensor] = None,
        position_reward: Optional[torch.Tensor] = None,
        src_in_idx: Optional[torch.Tensor] = None,
        tgt_in_idx: Optional[torch.Tensor] = None,
        tgt_out_idx: Optional[torch.Tensor] = None,
        tgt_out_probs: Optional[torch.Tensor] = None,
        optim_tgt_in_idx: Optional[torch.Tensor] = None,
        optim_tgt_out_idx: Optional[torch.Tensor] = None,
        optim_tgt_in_seq: Optional[torch.Tensor] = None,
        optim_tgt_out_seq: Optional[torch.Tensor] = None,
    ):
        assert isinstance(state, torch.Tensor)
        assert isinstance(src_seq, torch.Tensor)
        assert isinstance(src_src_mask, torch.Tensor)
        assert tgt_in_seq is None or isinstance(tgt_in_seq, torch.Tensor)
        assert tgt_out_seq is None or isinstance(tgt_out_seq, torch.Tensor)
        assert tgt_tgt_mask is None or isinstance(tgt_tgt_mask, torch.Tensor)
        assert slate_reward is None or isinstance(slate_reward, torch.Tensor)
        assert position_reward is None or isinstance(position_reward, torch.Tensor)
        assert src_in_idx is None or isinstance(src_in_idx, torch.Tensor)
        assert tgt_in_idx is None or isinstance(tgt_in_idx, torch.Tensor)
        assert tgt_out_idx is None or isinstance(tgt_out_idx, torch.Tensor)
        assert tgt_out_probs is None or isinstance(tgt_out_probs, torch.Tensor)
        assert optim_tgt_out_idx is None or isinstance(optim_tgt_out_idx, torch.Tensor)
        assert optim_tgt_out_idx is None or isinstance(optim_tgt_out_idx, torch.Tensor)
        assert optim_tgt_in_seq is None or isinstance(optim_tgt_in_seq, torch.Tensor)
        assert optim_tgt_out_seq is None or isinstance(optim_tgt_out_seq, torch.Tensor)

        return cls(
            state=PreprocessedFeatureVector(float_features=state),
            src_seq=PreprocessedFeatureVector(float_features=src_seq),
            src_src_mask=src_src_mask,
            tgt_in_seq=PreprocessedFeatureVector(float_features=tgt_in_seq)
            if tgt_in_seq is not None
            else None,
            tgt_out_seq=PreprocessedFeatureVector(float_features=tgt_out_seq)
            if tgt_out_seq is not None
            else None,
            tgt_tgt_mask=tgt_tgt_mask,
            slate_reward=slate_reward,
            position_reward=position_reward,
            src_in_idx=src_in_idx,
            tgt_in_idx=tgt_in_idx,
            tgt_out_idx=tgt_out_idx,
            tgt_out_probs=tgt_out_probs,
            optim_tgt_in_idx=optim_tgt_in_idx,
            optim_tgt_out_idx=optim_tgt_out_idx,
            optim_tgt_in_seq=PreprocessedFeatureVector(float_features=optim_tgt_in_seq)
            if optim_tgt_in_seq is not None
            else None,
            optim_tgt_out_seq=PreprocessedFeatureVector(
                float_features=optim_tgt_out_seq
            )
            if optim_tgt_out_seq is not None
            else None,
        )

    def __post_init__(self):
        if (
            isinstance(self.state, torch.Tensor)
            or isinstance(self.src_seq, torch.Tensor)
            or isinstance(self.tgt_in_seq, torch.Tensor)
            or isinstance(self.tgt_out_seq, torch.Tensor)
            or isinstance(self.optim_tgt_in_seq, torch.Tensor)
            or isinstance(self.optim_tgt_out_seq, torch.Tensor)
        ):
            raise ValueError(
                f"Use from_tensors() {type(self.state)} {type(self.src_seq)} "
                f"{type(self.tgt_in_seq)} {type(self.tgt_out_seq)} "
                f"{type(self.optim_tgt_in_seq)} {type(self.optim_tgt_out_seq)} "
            )


@dataclass
class RawStateAction(BaseDataClass):
    state: FeatureVector
    action: FeatureVector


@dataclass
class CommonInput(BaseDataClass):
    """
    Base class for all inputs, both raw and preprocessed
    """

    reward: torch.Tensor
    time_diff: torch.Tensor
    step: Optional[torch.Tensor]
    not_terminal: torch.Tensor


@dataclass
class PreprocessedBaseInput(CommonInput):
    state: PreprocessedFeatureVector
    next_state: PreprocessedFeatureVector


@dataclass
class PreprocessedDiscreteDqnInput(PreprocessedBaseInput):
    action: torch.Tensor
    next_action: torch.Tensor
    possible_actions_mask: torch.Tensor
    possible_next_actions_mask: torch.Tensor


@dataclass
class PreprocessedSlateFeatureVector(BaseDataClass):
    """
    The shape of `float_features` is
    `(batch_size, slate_size, item_dim)`.

    `item_mask` masks available items in the action

    `item_probability` is the probability of item in being selected
    """

    float_features: torch.Tensor
    item_mask: torch.Tensor
    item_probability: torch.Tensor

    def as_preprocessed_feature_vector(self) -> PreprocessedFeatureVector:
        return PreprocessedFeatureVector(
            float_features=self.float_features.view(-1, self.float_features.shape[2])
        )


@dataclass
class PreprocessedSlateQInput(PreprocessedBaseInput):
    """
    The shapes of `tiled_state` & `tiled_next_state` are
    `(batch_size, slate_size, state_dim)`.

    The shapes of `reward`, `reward_mask`, & `next_item_mask` are
    `(batch_size, slate_size)`.

    `reward_mask` indicated whether the reward could be observed, e.g.,
    the item got into viewport or not.
    """

    tiled_state: PreprocessedTiledFeatureVector
    tiled_next_state: PreprocessedTiledFeatureVector
    action: PreprocessedSlateFeatureVector
    next_action: PreprocessedSlateFeatureVector
    reward_mask: torch.Tensor


@dataclass
class PreprocessedParametricDqnInput(PreprocessedBaseInput):
    action: PreprocessedFeatureVector
    next_action: PreprocessedFeatureVector
    possible_actions: PreprocessedFeatureVector
    possible_actions_mask: torch.Tensor
    possible_next_actions: PreprocessedFeatureVector
    possible_next_actions_mask: torch.Tensor
    tiled_next_state: PreprocessedFeatureVector


@dataclass
class PreprocessedPolicyNetworkInput(PreprocessedBaseInput):
    action: PreprocessedFeatureVector
    next_action: PreprocessedFeatureVector


@dataclass
class PreprocessedMemoryNetworkInput(PreprocessedBaseInput):
    action: Union[torch.Tensor, torch.Tensor]


@dataclass
class RawBaseInput(CommonInput):
    state: FeatureVector
    next_state: FeatureVector


@dataclass
class RawDiscreteDqnInput(RawBaseInput):
    action: torch.Tensor
    next_action: torch.Tensor
    possible_actions_mask: torch.Tensor
    possible_next_actions_mask: torch.Tensor

    def preprocess(
        self, state: PreprocessedFeatureVector, next_state: PreprocessedFeatureVector
    ):
        assert isinstance(state, PreprocessedFeatureVector)
        assert isinstance(next_state, PreprocessedFeatureVector)
        return PreprocessedDiscreteDqnInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal.float(),
            state,
            next_state,
            self.action.float(),
            self.next_action.float(),
            self.possible_actions_mask.float(),
            self.possible_next_actions_mask.float(),
        )

    def preprocess_tensors(self, state: torch.Tensor, next_state: torch.Tensor):
        assert isinstance(state, torch.Tensor)
        assert isinstance(next_state, torch.Tensor)
        return PreprocessedDiscreteDqnInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal.float(),
            PreprocessedFeatureVector(
                float_features=state,
                id_list_features=self.state.id_list_features,
                time_since_first=self.state.time_since_first,
            ),
            PreprocessedFeatureVector(
                float_features=next_state,
                id_list_features=self.next_state.id_list_features,
                time_since_first=self.next_state.time_since_first,
            ),
            self.action.float(),
            self.next_action.float(),
            self.possible_actions_mask.float(),
            self.possible_next_actions_mask.float(),
        )


@dataclass
class RawParametricDqnInput(RawBaseInput):
    action: FeatureVector
    next_action: FeatureVector
    possible_actions: FeatureVector
    possible_actions_mask: torch.Tensor
    possible_next_actions: FeatureVector
    possible_next_actions_mask: torch.Tensor
    tiled_next_state: FeatureVector

    def preprocess(
        self,
        state: PreprocessedFeatureVector,
        next_state: PreprocessedFeatureVector,
        action: PreprocessedFeatureVector,
        next_action: PreprocessedFeatureVector,
        possible_actions: PreprocessedFeatureVector,
        possible_next_actions: PreprocessedFeatureVector,
        tiled_next_state: PreprocessedFeatureVector,
    ):
        assert isinstance(state, PreprocessedFeatureVector)
        assert isinstance(next_state, PreprocessedFeatureVector)
        assert isinstance(action, PreprocessedFeatureVector)
        assert isinstance(next_action, PreprocessedFeatureVector)
        assert isinstance(possible_actions, PreprocessedFeatureVector)
        assert isinstance(possible_next_actions, PreprocessedFeatureVector)
        assert isinstance(tiled_next_state, PreprocessedFeatureVector)

        return PreprocessedParametricDqnInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal,
            state,
            next_state,
            action,
            next_action,
            possible_actions,
            self.possible_actions_mask,
            possible_next_actions,
            self.possible_next_actions_mask,
            tiled_next_state,
        )

    def preprocess_tensors(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        next_action: torch.Tensor,
        possible_actions: torch.Tensor,
        possible_next_actions: torch.Tensor,
        tiled_next_state: torch.Tensor,
    ):
        assert isinstance(state, torch.Tensor)
        assert isinstance(next_state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(next_action, torch.Tensor)
        assert isinstance(possible_actions, torch.Tensor)
        assert isinstance(possible_next_actions, torch.Tensor)
        assert isinstance(tiled_next_state, torch.Tensor)
        return PreprocessedParametricDqnInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal,
            PreprocessedFeatureVector(float_features=state),
            PreprocessedFeatureVector(float_features=next_state),
            PreprocessedFeatureVector(float_features=action),
            PreprocessedFeatureVector(float_features=next_action),
            PreprocessedFeatureVector(float_features=possible_actions),
            self.possible_actions_mask,
            PreprocessedFeatureVector(float_features=possible_next_actions),
            self.possible_next_actions_mask,
            PreprocessedFeatureVector(float_features=tiled_next_state),
        )


@dataclass
class RawPolicyNetworkInput(RawBaseInput):
    action: FeatureVector
    next_action: FeatureVector

    def preprocess(
        self,
        state: PreprocessedFeatureVector,
        next_state: PreprocessedFeatureVector,
        action: PreprocessedFeatureVector,
        next_action: PreprocessedFeatureVector,
    ):
        assert isinstance(state, PreprocessedFeatureVector)
        assert isinstance(next_state, PreprocessedFeatureVector)
        assert isinstance(action, PreprocessedFeatureVector)
        assert isinstance(next_action, PreprocessedFeatureVector)
        return PreprocessedPolicyNetworkInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal,
            state,
            next_state,
            action,
            next_action,
        )

    def preprocess_tensors(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        next_action: torch.Tensor,
    ):
        assert isinstance(state, torch.Tensor)
        assert isinstance(next_state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(next_action, torch.Tensor)
        return PreprocessedPolicyNetworkInput(
            self.reward,
            self.time_diff,
            self.step,
            self.not_terminal,
            PreprocessedFeatureVector(float_features=state),
            PreprocessedFeatureVector(float_features=next_state),
            PreprocessedFeatureVector(float_features=action),
            PreprocessedFeatureVector(float_features=next_action),
        )


@dataclass
class RawMemoryNetworkInput(RawBaseInput):
    action: Union[FeatureVector, torch.Tensor]

    def preprocess(
        self,
        state: PreprocessedFeatureVector,
        next_state: PreprocessedFeatureVector,
        action: Optional[torch.Tensor] = None,
    ):
        assert isinstance(state, PreprocessedFeatureVector)
        assert isinstance(next_state, PreprocessedFeatureVector)
        if action is not None:
            assert isinstance(action, torch.Tensor)
            return PreprocessedMemoryNetworkInput(
                self.reward,
                self.time_diff,
                self.step,
                self.not_terminal,
                state,
                next_state,
                action,
            )
        else:
            assert isinstance(self.action, torch.Tensor)
            assert self.action.dtype == torch.uint8
            return PreprocessedMemoryNetworkInput(
                self.reward,
                self.time_diff,
                self.step,
                self.not_terminal,
                state,
                next_state,
                self.action.float(),
            )

    def preprocess_tensors(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        assert isinstance(state, torch.Tensor)
        assert isinstance(next_state, torch.Tensor)

        if action is not None:
            assert isinstance(action, torch.Tensor)
            return PreprocessedMemoryNetworkInput(
                self.reward,
                self.time_diff,
                self.step,
                self.not_terminal,
                PreprocessedFeatureVector(float_features=state),
                PreprocessedFeatureVector(float_features=next_state),
                action,
            )
        else:
            assert isinstance(self.action, torch.Tensor)
            assert self.action.dtype == torch.uint8
            return PreprocessedMemoryNetworkInput(
                self.reward,
                self.time_diff,
                self.step,
                self.not_terminal,
                PreprocessedFeatureVector(float_features=state),
                PreprocessedFeatureVector(float_features=next_state),
                self.action.float(),
            )


@dataclass
class ExtraData(BaseDataClass):
    mdp_id: Optional[
        np.ndarray
    ] = None  # Need to use a numpy array because torch doesn't support strings
    sequence_number: Optional[torch.Tensor] = None
    action_probability: Optional[torch.Tensor] = None
    max_num_actions: Optional[int] = None
    metrics: Optional[torch.Tensor] = None


@dataclass
class PreprocessedTrainingBatch(BaseDataClass):
    training_input: Union[
        PreprocessedBaseInput,
        PreprocessedDiscreteDqnInput,
        PreprocessedParametricDqnInput,
        PreprocessedMemoryNetworkInput,
        PreprocessedPolicyNetworkInput,
        PreprocessedRankingInput,
        PreprocessedSlateQInput,
    ]
    extras: Any

    def batch_size(self):
        return self.training_input.state.float_features.size()[0]


@dataclass
class RawTrainingBatch(BaseDataClass):
    training_input: Union[
        RawBaseInput, RawDiscreteDqnInput, RawParametricDqnInput, RawPolicyNetworkInput
    ]
    extras: Any

    def batch_size(self):
        return self.training_input.state.float_features.value.size()[0]

    def preprocess(
        self,
        training_input: Union[
            PreprocessedBaseInput,
            PreprocessedDiscreteDqnInput,
            PreprocessedParametricDqnInput,
            PreprocessedMemoryNetworkInput,
            PreprocessedPolicyNetworkInput,
        ],
    ) -> PreprocessedTrainingBatch:
        return PreprocessedTrainingBatch(
            training_input=training_input, extras=self.extras
        )


@dataclass
class SingleQValue(BaseDataClass):
    q_value: torch.Tensor


@dataclass
class AllActionQValues(BaseDataClass):
    q_values: torch.Tensor


@dataclass
class MemoryNetworkOutput(BaseDataClass):
    mus: torch.Tensor
    sigmas: torch.Tensor
    logpi: torch.Tensor
    reward: torch.Tensor
    not_terminal: torch.Tensor
    last_step_lstm_hidden: torch.Tensor
    last_step_lstm_cell: torch.Tensor
    all_steps_lstm_hidden: torch.Tensor


@dataclass
class DqnPolicyActionSet(BaseDataClass):
    greedy: int
    softmax: Optional[int] = None
    greedy_act_name: Optional[str] = None
    softmax_act_name: Optional[str] = None
    softmax_act_prob: Optional[float] = None


@dataclass
class SacPolicyActionSet(BaseDataClass):
    greedy: torch.Tensor
    greedy_propensity: float


@dataclass
class PlanningPolicyOutput(BaseDataClass):
    # best action to take next
    next_best_continuous_action: Optional[torch.Tensor] = None
    next_best_discrete_action_one_hot: Optional[torch.Tensor] = None
    next_best_discrete_action_idx: Optional[int] = None


@dataclass
class RankingOutput(BaseDataClass):
    # a tensor of integer indices w.r.t. to possible candidates
    # shape: batch_size, tgt_seq_len
    ranked_tgt_out_idx: Optional[torch.Tensor] = None
    # generative probability of ranked tgt sequences at each decoding step
    # shape: batch_size, tgt_seq_len, candidate_size
    ranked_tgt_out_probs: Optional[torch.Tensor] = None
    # log probabilities of given tgt sequences are used in REINFORCE
    # shape: batch_size
    log_probs: Optional[torch.Tensor] = None
    # encoder scores in tgt_out_idx order
    encoder_scores: Optional[torch.Tensor] = None


@dataclass
class RewardNetworkOutput(BaseDataClass):
    predicted_reward: torch.Tensor
