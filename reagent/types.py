#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
import logging

# The dataclasses in this file should be vanilla dataclass to have minimal overhead
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from reagent.core.dataclasses import dataclass as pydantic_dataclass


class NoDuplicatedWarningLogger:
    def __init__(self, logger):
        self.logger = logger
        self.msg = set([])

    def warning(self, msg):
        if msg not in self.msg:
            self.logger.warning(msg)
            self.msg.add(msg)


logger = logging.getLogger(__name__)
no_dup_logger = NoDuplicatedWarningLogger(logger)


def isinstance_namedtuple(x):
    return isinstance(x, tuple) and hasattr(x, "_fields")


"""
We should revisit this at some point. Config classes shouldn't subclass from this.
"""


@dataclass
class BaseDataClass:
    def _replace(self, **kwargs):
        return cast(type(self), dataclasses.replace(self, **kwargs))


@dataclass
class TensorDataClass(BaseDataClass):
    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError

        tensor_attr = getattr(torch.Tensor, attr, None)

        if tensor_attr is None or not callable(tensor_attr):
            logger.error(
                f"Attemping to call torch.Tensor.{attr} on "
                f"{type(self)} (instance of TensorDataClass)."
            )
            if tensor_attr is None:
                raise AttributeError(f"torch.Tensor doesn't have {attr} attribute.")
            else:
                raise RuntimeError(f"Tensor.{attr} is not callable.")

        def f(*args, **kwargs):
            values = {}
            for k, v in self.__dict__.items():  # noqa F402
                if isinstance(v, (torch.Tensor, TensorDataClass)):
                    values[k] = getattr(v, attr)(*args, **kwargs)
                else:
                    values[k] = v
            return type(self)(**values)

        return f

    def cuda(self, *args, **kwargs):
        cuda_tensor = {}
        for k, v in self.__dict__.items():  # noqa F402
            if isinstance(v, torch.Tensor):
                kwargs["non_blocking"] = kwargs.get("non_blocking", True)
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            elif isinstance(v, TensorDataClass):
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            else:
                cuda_tensor[k] = v
        return type(self)(**cuda_tensor)


#####
# FIXME: These config types are misplaced but we need to write FBL config adapter
# if we moved them.
######


@pydantic_dataclass
class IdListFeatureConfig(BaseDataClass):
    """
    This describes how to map raw features to model features
    """

    name: str
    feature_id: int  # integer feature ID
    id_mapping_name: str  # key to ModelPreprocessingConfig.id_mapping_config
    # max_length: int


@pydantic_dataclass
class FloatFeatureInfo(BaseDataClass):
    name: str
    feature_id: int


@pydantic_dataclass
class IdMapping(BaseDataClass):
    ids: List[int]


@pydantic_dataclass
class ModelFeatureConfig(BaseDataClass):
    float_feature_infos: List[FloatFeatureInfo]
    id_mapping_config: Dict[str, IdMapping] = field(default_factory=dict)
    id_list_feature_configs: List[IdListFeatureConfig] = field(default_factory=list)


######
# dataclasses for internal API
######


@dataclass
class ValuePresence(TensorDataClass):
    value: torch.Tensor
    presence: Optional[torch.Tensor]


IdListFeatureValue = Tuple[torch.Tensor, torch.Tensor]
IdListFeatures = Dict[str, IdListFeatureValue]


@dataclass
class RawFeatureData(TensorDataClass):
    float_features: ValuePresence
    id_list_features: IdListFeatures = dataclasses.field(default_factory=dict)
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[torch.Tensor] = None


@dataclass
class ActorOutput(TensorDataClass):
    action: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    action_mean: Optional[torch.Tensor] = None


@dataclass
class DocList(TensorDataClass):
    # the shape is (batch_size, num_candidates, num_document_features)
    float_features: torch.Tensor
    # the shapes are (batch_size, num_candidates)
    mask: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        assert (
            len(self.float_features.shape) == 3
        ), f"Unexpected shape: {self.float_features.shape}"

    @torch.no_grad()
    def select_slate(self, action: torch.Tensor):
        row_idx = torch.repeat_interleave(
            torch.arange(action.shape[0]).unsqueeze(1), action.shape[1], dim=1
        )
        mask = self.mask[row_idx, action]
        # Make sure the indices are in the right range
        assert mask.to(torch.bool).all()
        float_features = self.float_features[row_idx, action]
        value = self.value[row_idx, action]
        return DocList(float_features, mask, value)

    def as_feature_data(self):
        _batch_size, _slate_size, feature_dim = self.float_features.shape
        return FeatureData(self.float_features.view(-1, feature_dim))


@dataclass
class FeatureData(TensorDataClass):
    # For dense features, shape is (batch_size, feature_dim)
    float_features: torch.Tensor
    # For sequence, shape is (stack_size, batch_size, feature_dim)
    stacked_float_features: Optional[torch.Tensor] = None
    id_list_features: IdListFeatures = dataclasses.field(default_factory=dict)
    # For ranking algos,
    candidate_docs: Optional[DocList] = None
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[torch.Tensor] = None

    def __post_init__(self):
        def usage():
            return (
                f"For sequence features, use `stacked_float_features`."
                f"For document features, use `candidate_doc_float_features`."
            )

        if self.float_features.ndim == 3:
            no_dup_logger.warning(f"`float_features` should be 2D.\n{usage()}")
        elif self.float_features.ndim != 2:
            raise ValueError(
                f"float_features should be 2D; got {self.float_features.shape}.\n{usage()}"
            )

    @classmethod
    def from_raw_feature_data(cls, feature_vector: RawFeatureData, preprocessor):
        return cls(
            float_features=preprocessor(
                feature_vector.float_features.value,
                feature_vector.float_features.presence,
            ),
            id_list_features=feature_vector.id_list_features,
            time_since_first=feature_vector.time_since_first,
        )

    @classmethod
    def from_dict(cls, d, name: str):
        # TODO: Looks for id_list_features
        return cls(float_features=d[name])

    @property
    def has_float_features_only(self) -> bool:
        return (
            not self.id_list_features
            and self.time_since_first is None
            and self.candidate_docs is None
        )

    def get_tiled_batch(self, num_tiles: int):
        assert (
            self.has_float_features_only
        ), f"only works for float features now: {self}"
        """
        tiled_feature should be (batch_size * num_tiles, feature_dim)
        forall i in [batch_size],
        tiled_feature[i*num_tiles:(i+1)*num_tiles] should be feat[i]
        """
        feat = self.float_features
        assert (
            len(feat.shape) == 2
        ), f"Need feat shape to be (batch_size, feature_dim), got {feat.shape}."
        batch_size, _ = feat.shape
        # pyre-fixme[16]: `Tensor` has no attribute `repeat_interleave`.
        tiled_feat = feat.repeat_interleave(repeats=num_tiles, dim=0)
        return FeatureData(float_features=tiled_feat)


class TensorFeatureData(torch.nn.Module):
    """
    Primarily for using in nn.Sequential
    """

    def forward(self, input: torch.Tensor) -> FeatureData:
        assert isinstance(input, torch.Tensor)
        return FeatureData(input)


@dataclass
class PreprocessedRankingInput(TensorDataClass):
    state: FeatureData
    src_seq: FeatureData
    src_src_mask: torch.Tensor
    tgt_in_seq: Optional[FeatureData] = None
    tgt_out_seq: Optional[FeatureData] = None
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
    optim_tgt_in_seq: Optional[FeatureData] = None
    optim_tgt_out_seq: Optional[FeatureData] = None

    def batch_size(self) -> int:
        return self.state.float_features.size()[0]

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
        **kwargs,
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
            state=FeatureData(float_features=state),
            src_seq=FeatureData(float_features=src_seq),
            src_src_mask=src_src_mask,
            tgt_in_seq=FeatureData(float_features=tgt_in_seq)
            if tgt_in_seq is not None
            else None,
            tgt_out_seq=FeatureData(float_features=tgt_out_seq)
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
            optim_tgt_in_seq=FeatureData(float_features=optim_tgt_in_seq)
            if optim_tgt_in_seq is not None
            else None,
            optim_tgt_out_seq=FeatureData(float_features=optim_tgt_out_seq)
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
class CommonInput(TensorDataClass):
    """
    Base class for all inputs, both raw and preprocessed
    """

    reward: torch.Tensor
    time_diff: torch.Tensor
    step: Optional[torch.Tensor]
    not_terminal: torch.Tensor


@dataclass
class ExtraData(TensorDataClass):
    mdp_id: Optional[torch.Tensor] = None
    sequence_number: Optional[torch.Tensor] = None
    action_probability: Optional[torch.Tensor] = None
    max_num_actions: Optional[int] = None
    metrics: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d.get(f.name, None) for f in dataclasses.fields(cls)})


@dataclass
class PreprocessedBaseInput(CommonInput):
    state: FeatureData
    next_state: FeatureData

    def batch_size(self):
        return self.state.float_features.size()[0]


@dataclass
class DiscreteDqnInput(PreprocessedBaseInput):
    action: torch.Tensor
    next_action: torch.Tensor
    possible_actions_mask: torch.Tensor
    possible_next_actions_mask: torch.Tensor
    extras: ExtraData


@dataclass
class SlateQInput(PreprocessedBaseInput):
    """
    The shapes of `reward`, `reward_mask`, & `next_item_mask` are
    `(batch_size, slate_size)`.

    `reward_mask` indicated whether the reward could be observed, e.g.,
    the item got into viewport or not.
    """

    action: torch.Tensor
    next_action: torch.Tensor
    reward_mask: torch.Tensor
    extras: Optional[ExtraData] = None

    @classmethod
    def from_dict(cls, d):
        action = d["action"]
        next_action = d["next_action"]
        return cls(
            state=FeatureData(
                float_features=d["state_features"],
                candidate_docs=DocList(
                    float_features=d["candidate_features"],
                    mask=d["item_mask"],
                    value=d["item_probability"],
                ),
            ),
            next_state=FeatureData(
                float_features=d["next_state_features"],
                candidate_docs=DocList(
                    float_features=d["next_candidate_features"],
                    mask=d["next_item_mask"],
                    value=d["next_item_probability"],
                ),
            ),
            action=action,
            next_action=next_action,
            reward=d["position_reward"],
            reward_mask=d["reward_mask"],
            time_diff=d["time_diff"],
            not_terminal=d["not_terminal"],
            step=None,
            extras=ExtraData.from_dict(d),
        )


@dataclass
class ParametricDqnInput(PreprocessedBaseInput):
    action: FeatureData
    next_action: FeatureData
    possible_actions: FeatureData
    possible_actions_mask: torch.Tensor
    possible_next_actions: FeatureData
    possible_next_actions_mask: torch.Tensor
    extras: Optional[ExtraData] = None

    @classmethod
    def from_dict(cls, batch):
        return cls(
            state=FeatureData(float_features=batch["state_features"]),
            action=FeatureData(float_features=batch["action"]),
            next_state=FeatureData(float_features=batch["next_state_features"]),
            next_action=FeatureData(float_features=batch["next_action"]),
            possible_actions=FeatureData(float_features=batch["possible_actions"]),
            possible_actions_mask=batch["possible_actions_mask"],
            possible_next_actions=FeatureData(
                float_features=batch["possible_next_actions"]
            ),
            possible_next_actions_mask=batch["possible_next_actions_mask"],
            reward=batch["reward"],
            not_terminal=batch["not_terminal"],
            time_diff=batch["time_diff"],
            step=batch["step"],
            extras=batch["extras"],
        )


@dataclass
class PolicyNetworkInput(PreprocessedBaseInput):
    action: FeatureData
    next_action: FeatureData
    extras: Optional[ExtraData] = None

    @classmethod
    def from_dict(cls, batch):
        return cls(
            state=FeatureData(float_features=batch["state_features"]),
            action=FeatureData(float_features=batch["action"]),
            next_state=FeatureData(float_features=batch["next_state_features"]),
            next_action=FeatureData(float_features=batch["next_action"]),
            reward=batch["reward"],
            not_terminal=batch["not_terminal"],
            time_diff=batch["time_diff"],
            step=batch["step"],
            extras=batch["extras"],
        )

    def batch_size(self) -> int:
        return self.state.float_features.shape[0]


# TODO(T67083627): state and next_state should use stack_float_features
@dataclass
class MemoryNetworkInput(PreprocessedBaseInput):
    action: torch.Tensor


@dataclass
class RawBaseInput(CommonInput):
    state: RawFeatureData
    next_state: RawFeatureData


@dataclass
class PreprocessedTrainingBatch(TensorDataClass):
    training_input: Union[PreprocessedRankingInput]
    # TODO: deplicate this and move into individual ones.
    extras: ExtraData = field(default_factory=ExtraData)

    def batch_size(self):
        return self.training_input.state.float_features.size()[0]


@dataclass
class MemoryNetworkOutput(TensorDataClass):
    mus: torch.Tensor
    sigmas: torch.Tensor
    logpi: torch.Tensor
    reward: torch.Tensor
    not_terminal: torch.Tensor
    last_step_lstm_hidden: torch.Tensor
    last_step_lstm_cell: torch.Tensor
    all_steps_lstm_hidden: torch.Tensor


@dataclass
class DqnPolicyActionSet(TensorDataClass):
    greedy: int
    softmax: Optional[int] = None
    greedy_act_name: Optional[str] = None
    softmax_act_name: Optional[str] = None
    softmax_act_prob: Optional[float] = None


@dataclass
class SacPolicyActionSet(TensorDataClass):
    greedy: torch.Tensor
    greedy_propensity: float


@dataclass
class PlanningPolicyOutput(TensorDataClass):
    # best action to take next
    next_best_continuous_action: Optional[torch.Tensor] = None
    next_best_discrete_action_one_hot: Optional[torch.Tensor] = None
    next_best_discrete_action_idx: Optional[int] = None


@dataclass
class RankingOutput(TensorDataClass):
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
class RewardNetworkOutput(TensorDataClass):
    predicted_reward: torch.Tensor
