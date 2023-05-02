#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
import logging

# The dataclasses in this file should be vanilla dataclass to have minimal overhead
from dataclasses import dataclass, field
from typing import Dict, Final, List, NamedTuple, Optional, Tuple

# Triggering registration to registries
import reagent.core.result_types  # noqa
import torch
import torch.nn.functional as F
from reagent.core.base_dataclass import BaseDataClass
from reagent.core.dataclasses import dataclass as pydantic_dataclass
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.torch_utils import gather
from reagent.model_utils.seq2slate_utils import DECODER_START_SYMBOL, subsequent_mask
from reagent.preprocessing.types import InputColumn
from torchrec import PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


if IS_FB_ENVIRONMENT:
    import reagent.core.fb.fb_result_types  # noqa


class NoDuplicatedWarningLogger:
    def __init__(self, logger):
        self.logger = logger
        self.msg = set()

    def warning(self, msg):
        if msg not in self.msg:
            self.logger.warning(msg)
            self.msg.add(msg)


logger = logging.getLogger(__name__)
no_dup_logger = NoDuplicatedWarningLogger(logger)


def isinstance_namedtuple(x):
    return isinstance(x, tuple) and hasattr(x, "_fields")


@dataclass
class TensorDataClass(BaseDataClass):
    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError

        tensor_attr = getattr(torch.Tensor, attr, None)

        if tensor_attr is None or not callable(tensor_attr):
            # TODO: can we get this working well with jupyter?
            logger.error(
                f"Attempting to call {self.__class__.__name__}.{attr} on "
                f"{type(self)} (instance of TensorDataClass)."
            )
            if tensor_attr is None:
                raise AttributeError(
                    f"{self.__class__.__name__}doesn't have {attr} attribute."
                )
            else:
                raise RuntimeError(f"{self.__class__.__name__}.{attr} is not callable.")

        def continuation(*args, **kwargs):
            def f(v):
                # if possible, returns v.attr(*args, **kwargs).
                # otws, return v
                if (
                    isinstance(v, (torch.Tensor, TensorDataClass, KeyedJaggedTensor))
                    and getattr(v, attr, None) is not None
                ):
                    return getattr(v, attr)(*args, **kwargs)
                elif isinstance(v, dict):
                    return {kk: f(vv) for kk, vv in v.items()}
                elif isinstance(v, tuple):
                    return tuple(f(vv) for vv in v)
                return v

            return type(self)(**f(self.__dict__))

        return continuation

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

    def cpu(self):
        cpu_tensor = {}
        for k, v in self.__dict__.items():  # noqa F402
            if isinstance(v, (torch.Tensor, TensorDataClass)):
                cpu_tensor[k] = v.cpu()
            else:
                cpu_tensor[k] = v
        return type(self)(**cpu_tensor)


# (offset, value)
IdListFeatureValue = Tuple[torch.Tensor, torch.Tensor]
# (offset, key, value)
IdScoreListFeatureValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# name -> value
IdListFeature = Dict[str, IdListFeatureValue]
IdScoreListFeature = Dict[str, IdScoreListFeatureValue]
# id -> value
ServingIdListFeature = Dict[int, IdListFeatureValue]
ServingIdScoreListFeature = Dict[int, IdScoreListFeatureValue]


#####
# FIXME: These config types are misplaced but we need to write FBL config adapter
# if we moved them.
######


@pydantic_dataclass
class IdListFeatureConfig(BaseDataClass):
    # Feature name
    name: str
    # integer feature ID
    feature_id: int
    # Name of the embedding table to use. Multiple feature ids may share
    # the same embedding table.
    id_mapping_name: str


@pydantic_dataclass
class IdScoreListFeatureConfig(BaseDataClass):
    # Feature name
    name: str
    # Integer feature ID
    feature_id: int
    # Name of the embedding table to use. Multiple feature ids may share
    # the same embedding table.
    id_mapping_name: str


@pydantic_dataclass
class FloatFeatureInfo(BaseDataClass):
    name: str
    feature_id: int


@pydantic_dataclass
class IdMappingConfig:
    # Embedding table size.
    embedding_table_size: int

    # Output embedding dimensions
    embedding_dim: int

    # Whether to perform hashing to make id fall in the range of embedding_table_size
    # If False, the user is at their own risk of raw ids going beyond the range
    hashing: bool = True

    pooling_type: PoolingType = PoolingType.MEAN

    def __eq__(self, other):
        return (
            self.embedding_table_size == other.embedding_table_size
            and self.embedding_dim == other.embedding_dim
            and self.hashing == other.hashing
            and self.pooling_type == other.pooling_type
        )


@pydantic_dataclass
class ModelFeatureConfig(BaseDataClass):
    float_feature_infos: List[FloatFeatureInfo] = field(default_factory=list)
    # id_mapping_name -> id mapping config
    id_mapping_config: Dict[str, IdMappingConfig] = field(default_factory=dict)
    # id_list_feature_configs is feature_id -> list of values
    id_list_feature_configs: List[IdListFeatureConfig] = field(default_factory=list)
    # id_score_list_feature_configs is feature_id -> (keys -> values)
    id_score_list_feature_configs: List[IdScoreListFeatureConfig] = field(
        default_factory=list
    )

    def __post_init_post_parse__(self):
        both_lists = self.id_list_feature_configs + self.id_score_list_feature_configs
        if not self.only_dense:
            # sanity check for keys in mapping config
            ids = [config.feature_id for config in both_lists]
            names = [config.name for config in both_lists]
            assert len(ids) == len(set(ids)), f"duplicates in ids: {ids}"
            assert len(names) == len(set(names)), f"duplicates in names: {names}"
            assert len(ids) == len(names), f"{len(ids)} != {len(names)}"
            id_mapping_names = [config.id_mapping_name for config in both_lists]
            assert set(id_mapping_names) == set(self.id_mapping_config.keys()), (
                f"id_mapping_names in id_list_feature_configs/id_score_list_feature_configs "
                f"({set(id_mapping_names)}) not match with those in "
                f"id_mapping_config ({set(self.id_mapping_config.keys())})"
            )

        self._id2name = {config.feature_id: config.name for config in both_lists}
        self._name2id = {config.name: config.feature_id for config in both_lists}
        self._id2config = {config.feature_id: config for config in both_lists}
        self._name2config = {config.name: config for config in both_lists}

    @property
    def only_dense(self):
        return not (self.id_list_feature_configs or self.id_score_list_feature_configs)

    @property
    def id2name(self):
        return self._id2name

    @property
    def name2id(self):
        return self._name2id

    @property
    def id2config(self):
        return self._id2config

    @property
    def name2config(self):
        return self._name2config


######
# dataclasses for internal API
######


@dataclass
class ValuePresence(TensorDataClass):
    value: torch.Tensor
    presence: Optional[torch.Tensor]


@dataclass
class ActorOutput(TensorDataClass):
    action: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    squashed_mean: Optional[torch.Tensor] = None


@dataclass
class DocList(TensorDataClass):
    # the shape is (batch_size, num_candidates, num_document_features)
    float_features: torch.Tensor
    # the shapes below are (batch_size, num_candidates)
    # mask indicates whether the candidate is present or not; its dtype is torch.bool
    # pyre-fixme[8]: Attribute has type `Tensor`; used as `None`.
    mask: torch.Tensor = None
    # value is context dependent; it could be action probability or the score
    # of the document from another model
    # pyre-fixme[8]: Attribute has type `Tensor`; used as `None`.
    value: torch.Tensor = None

    def __post_init__(self):
        assert (
            len(self.float_features.shape) == 3
        ), f"Unexpected shape: {self.float_features.shape}"
        if self.mask is None:
            self.mask = self.float_features.new_ones(
                self.float_features.shape[:2], dtype=torch.bool
            )
        if self.value is None:
            self.value = self.float_features.new_ones(self.float_features.shape[:2])

    @torch.no_grad()
    def select_slate(self, action: torch.Tensor):
        row_idx = torch.repeat_interleave(
            torch.arange(action.shape[0]).unsqueeze(1), action.shape[1], dim=1
        )
        mask = self.mask[row_idx, action]
        float_features = self.float_features[row_idx, action]
        value = self.value[row_idx, action]
        return DocList(float_features, mask, value)

    def as_feature_data(self):
        _batch_size, _slate_size, feature_dim = self.float_features.shape
        return FeatureData(self.float_features.view(-1, feature_dim))


# This method contains dynamic control flow
# Use torch.fx.wrap to mark it as a leaf module for FX tracing
@torch.fx.wrap
def run_post_init_validation(
    float_features: torch.Tensor,
) -> None:
    usage: str = (
        "For sequence features, use `stacked_float_features`."
        + "For document features, use `candidate_doc_float_features`."
    )

    if float_features.ndim == 3:
        if not torch.jit.is_scripting():
            no_dup_logger.warning(f"`float_features` should be 2D.\n{usage}")
        pass
    elif float_features.ndim != 2:
        raise ValueError(
            f"float_features should be 2D; got {float_features.shape}.\n{usage}"
        )


@dataclass
class FeatureData(TensorDataClass):
    # For dense features, shape is (batch_size, feature_dim)
    float_features: torch.Tensor
    # For sparse features saved in KeyedJaggedTensor format
    id_list_features: Optional[KeyedJaggedTensor] = None
    id_score_list_features: Optional[KeyedJaggedTensor] = None

    # For sparse features saved in dictionary format
    id_list_features_raw: IdListFeature = dataclasses.field(default_factory=dict)
    id_score_list_features_raw: IdScoreListFeature = dataclasses.field(
        default_factory=dict
    )

    # For sequence, shape is (stack_size, batch_size, feature_dim)
    stacked_float_features: Optional[torch.Tensor] = None
    # For ranking algos,
    candidate_docs: Optional[DocList] = None
    # Experimental: sticking this here instead of putting it in float_features
    # because a lot of places derive the shape of float_features from
    # normalization parameters.
    time_since_first: Optional[torch.Tensor] = None

    def __post_init__(self):
        run_post_init_validation(
            float_features=self.float_features,
        )

    @property
    def has_float_features_only(self) -> bool:
        return (
            not self.id_list_features
            and not self.id_score_list_features
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
        tiled_feat = feat.repeat_interleave(repeats=num_tiles, dim=0)
        return FeatureData(float_features=tiled_feat)

    def concat_user_doc(self):
        assert not self.has_float_features_only, "only works when DocList present"
        assert self.float_features.dim() == 2  # batch_size x state_dim
        batch_size, state_dim = self.float_features.shape
        # batch_size x num_docs x candidate_dim
        assert self.candidate_docs.float_features.dim() == 3
        assert len(self.candidate_docs.float_features) == batch_size
        _, num_docs, candidate_dim = self.candidate_docs.float_features.shape
        state_tiled = (
            torch.repeat_interleave(self.float_features, num_docs, dim=0)
            .reshape(batch_size, num_docs, state_dim)
            .float()
        )
        return torch.cat((state_tiled, self.candidate_docs.float_features), dim=2)

    def get_ranking_state(self, has_user_feat: bool):
        if has_user_feat:
            return self.concat_user_doc()
        else:
            # pyre-fixme[16]: `Optional` has no attribute `float_features`.
            return self.candidate_docs.float_features.float()


def _embed_states(x: FeatureData) -> FeatureData:
    """
    Get dense feature from float and doc features.
    TODO: make this an embedder.
    """
    assert x.candidate_docs is not None

    def _concat_state_candidates(state: torch.Tensor, candidates: torch.Tensor):
        """
        Expect
        state.shape = (n, state_dim),
        candidate.shape = (n, num_candidates, candidate_dim),

        Result has shape (n, state_dim + candidate_dim)
        [state, mean of candidates]
        """
        n = state.shape[0]
        assert len(state.shape) == 2, f"{state.shape} != (batch_size, user_dim)"
        assert (
            len(candidates.shape) == 3
        ), f"{candidates.shape} != (batch_size, num_candidates, candidate_dim)"
        assert candidates.shape[0] == n, f"{candidates.shape} 0th dim != {n}"
        # TODO: have an embedder here
        # NOTE: mean aggregation is not very effective here
        candidates_embedding = candidates.view(n, -1)
        return torch.cat([state, candidates_embedding], dim=1)

    return FeatureData(
        float_features=_concat_state_candidates(
            x.float_features,
            x.candidate_docs.float_features,
        )
    )


class TensorFeatureData(torch.nn.Module):
    """
    Primarily for using in nn.Sequential
    """

    def forward(self, input: torch.Tensor) -> FeatureData:
        assert isinstance(input, torch.Tensor)
        return FeatureData(input)


class ServingFeatureData(NamedTuple):
    float_features_with_presence: Tuple[torch.Tensor, torch.Tensor]
    id_list_features: ServingIdListFeature
    id_score_list_features: ServingIdScoreListFeature


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
class PreprocessedRankingInput(TensorDataClass):
    state: FeatureData
    src_seq: FeatureData
    src_src_mask: Optional[torch.Tensor] = None
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
    extras: Optional[ExtraData] = field(default_factory=ExtraData)

    def batch_size(self) -> int:
        return self.state.float_features.size()[0]

    def __len__(self) -> int:
        return self.batch_size()

    @classmethod
    def from_input(
        cls,
        state: torch.Tensor,
        candidates: torch.Tensor,
        device: torch.device,
        action: Optional[torch.Tensor] = None,
        optimal_action: Optional[torch.Tensor] = None,
        logged_propensities: Optional[torch.Tensor] = None,
        slate_reward: Optional[torch.Tensor] = None,
        position_reward: Optional[torch.Tensor] = None,
        extras: Optional[ExtraData] = None,
    ):
        """
        Build derived fields (indices & masks) from raw input
        """
        # Shape checking
        assert len(state.shape) == 2
        assert len(candidates.shape) == 3
        state = state.to(device)
        candidates = candidates.to(device)

        if action is not None:
            assert len(action.shape) == 2
            action = action.to(device)
        if logged_propensities is not None:
            assert (
                len(logged_propensities.shape) == 2
                and logged_propensities.shape[1] == 1
            )
            logged_propensities = logged_propensities.to(device)

        batch_size, candidate_num, candidate_dim = candidates.shape
        if slate_reward is not None:
            assert len(slate_reward.shape) == 2 and slate_reward.shape[1] == 1
            slate_reward = slate_reward.to(device)
        if position_reward is not None:
            # pyre-fixme[16]: `Optional` has no attribute `shape`.
            assert position_reward.shape == action.shape
            position_reward = position_reward.to(device)

        src_in_idx = (
            torch.arange(candidate_num, device=device).repeat(batch_size, 1) + 2
        )
        src_src_mask = (
            (torch.ones(batch_size, candidate_num, candidate_num))
            .type(torch.int8)
            .to(device)
        )

        def process_tgt_seq(action):
            if action is not None:
                _, output_size = action.shape
                # Account for decoder starting symbol and padding symbol
                candidates_augment = torch.cat(
                    (
                        torch.zeros(batch_size, 2, candidate_dim, device=device),
                        candidates,
                    ),
                    dim=1,
                )
                tgt_out_idx = action + 2
                tgt_in_idx = torch.full(
                    (batch_size, output_size), DECODER_START_SYMBOL, device=device
                )
                tgt_in_idx[:, 1:] = tgt_out_idx[:, :-1]
                tgt_out_seq = gather(candidates_augment, tgt_out_idx)
                tgt_in_seq = torch.zeros(
                    batch_size, output_size, candidate_dim, device=device
                )
                tgt_in_seq[:, 1:] = tgt_out_seq[:, :-1]
                tgt_tgt_mask = subsequent_mask(output_size, device)
            else:
                tgt_in_idx = None
                tgt_out_idx = None
                tgt_in_seq = None
                tgt_out_seq = None
                tgt_tgt_mask = None

            return tgt_in_idx, tgt_out_idx, tgt_in_seq, tgt_out_seq, tgt_tgt_mask

        (
            tgt_in_idx,
            tgt_out_idx,
            tgt_in_seq,
            tgt_out_seq,
            tgt_tgt_mask,
        ) = process_tgt_seq(action)
        (
            optim_tgt_in_idx,
            optim_tgt_out_idx,
            optim_tgt_in_seq,
            optim_tgt_out_seq,
            _,
        ) = process_tgt_seq(optimal_action)

        return cls.from_tensors(
            state=state,
            src_seq=candidates,
            src_src_mask=src_src_mask,
            tgt_in_seq=tgt_in_seq,
            tgt_out_seq=tgt_out_seq,
            tgt_tgt_mask=tgt_tgt_mask,
            slate_reward=slate_reward,
            position_reward=position_reward,
            src_in_idx=src_in_idx,
            tgt_in_idx=tgt_in_idx,
            tgt_out_idx=tgt_out_idx,
            tgt_out_probs=logged_propensities,
            optim_tgt_in_idx=optim_tgt_in_idx,
            optim_tgt_out_idx=optim_tgt_out_idx,
            optim_tgt_in_seq=optim_tgt_in_seq,
            optim_tgt_out_seq=optim_tgt_out_seq,
            extras=extras,
        )

    @classmethod
    def from_tensors(
        cls,
        state: torch.Tensor,
        src_seq: torch.Tensor,
        src_src_mask: Optional[torch.Tensor] = None,
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
        extras: Optional[ExtraData] = None,
        **kwargs,
    ):
        assert isinstance(state, torch.Tensor)
        assert isinstance(src_seq, torch.Tensor)
        assert src_src_mask is None or isinstance(src_src_mask, torch.Tensor)
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
        assert extras is None or isinstance(extras, ExtraData)

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
            extras=extras if extras is not None else None,
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
class BaseInput(TensorDataClass):
    """
    Base class for all inputs, both raw and preprocessed
    """

    state: FeatureData
    next_state: FeatureData
    reward: torch.Tensor
    time_diff: torch.Tensor
    step: Optional[torch.Tensor]
    not_terminal: torch.Tensor

    def __len__(self):
        assert self.state.float_features.ndim == 2
        return self.state.float_features.size()[0]

    def batch_size(self):
        return len(self)

    def as_dict_shallow(self):
        return {
            "state": self.state,
            "next_state": self.next_state,
            "reward": self.reward,
            "time_diff": self.time_diff,
            "step": self.step,
            "not_terminal": self.not_terminal,
        }

    @staticmethod
    def from_dict(batch):
        id_list_features = batch.get(InputColumn.STATE_ID_LIST_FEATURES, None)
        id_score_list_features = batch.get(
            InputColumn.STATE_ID_SCORE_LIST_FEATURES, None
        )
        next_id_list_features = batch.get(InputColumn.NEXT_STATE_ID_LIST_FEATURES, None)
        next_id_score_list_features = batch.get(
            InputColumn.NEXT_STATE_ID_SCORE_LIST_FEATURES, None
        )
        # TODO: handle value/mask of DocList
        filler_mask_val = None
        doc_list = None
        candidate_features = batch.get(InputColumn.CANDIDATE_FEATURES, None)
        if candidate_features is not None:
            filler_mask_val = torch.zeros(
                (candidate_features.shape[0], candidate_features.shape[1])
            )
            doc_list = DocList(
                float_features=candidate_features,
                mask=filler_mask_val.clone().bool(),
                value=filler_mask_val.clone().float(),
            )

        next_doc_list = None
        next_candidate_features = batch.get(InputColumn.NEXT_CANDIDATE_FEATURES, None)
        if next_candidate_features is not None:
            assert filler_mask_val is not None
            next_doc_list = DocList(
                float_features=next_candidate_features,
                mask=filler_mask_val.clone().bool(),
                value=filler_mask_val.clone().float(),
            )

        return BaseInput(
            state=FeatureData(
                float_features=batch[InputColumn.STATE_FEATURES],
                id_list_features=id_list_features,
                id_score_list_features=id_score_list_features,
                candidate_docs=doc_list,
            ),
            next_state=FeatureData(
                float_features=batch[InputColumn.NEXT_STATE_FEATURES],
                id_list_features=next_id_list_features,
                id_score_list_features=next_id_score_list_features,
                candidate_docs=next_doc_list,
            ),
            reward=batch[InputColumn.REWARD],
            time_diff=batch[InputColumn.TIME_DIFF],
            step=batch.get(InputColumn.STEP, None),
            not_terminal=batch[InputColumn.NOT_TERMINAL],
        )


@dataclass
class DiscreteDqnInput(BaseInput):
    """
    See input_prototype for DQN expected input shapes
    """

    action: torch.Tensor
    next_action: torch.Tensor
    possible_actions_mask: torch.Tensor
    possible_next_actions_mask: torch.Tensor
    extras: ExtraData

    @classmethod
    def input_prototype(cls, action_dim=2, batch_size=10, state_dim=3):
        return cls(
            state=FeatureData(float_features=torch.randn(batch_size, state_dim)),
            next_state=FeatureData(float_features=torch.randn(batch_size, state_dim)),
            reward=torch.rand(batch_size, 1),
            time_diff=torch.ones(batch_size, 1),
            step=torch.ones(batch_size, 1),
            not_terminal=torch.ones(batch_size, 1),
            action=F.one_hot(
                torch.randint(high=action_dim, size=(batch_size,)),
                num_classes=action_dim,
            ),
            next_action=F.one_hot(
                torch.randint(high=action_dim, size=(batch_size,)),
                num_classes=action_dim,
            ),
            possible_actions_mask=torch.ones(batch_size, action_dim),
            possible_next_actions_mask=torch.ones(batch_size, action_dim),
            extras=ExtraData(action_probability=torch.ones(batch_size, 1)),
        )

    @classmethod
    def from_dict(cls, batch):
        base = super().from_dict(batch)
        return cls(
            action=batch[InputColumn.ACTION],
            next_action=batch[InputColumn.NEXT_ACTION],
            possible_actions_mask=batch[InputColumn.POSSIBLE_ACTIONS_MASK],
            possible_next_actions_mask=batch[InputColumn.POSSIBLE_NEXT_ACTIONS_MASK],
            extras=ExtraData.from_dict(batch),
            **base.as_dict_shallow(),
        )


@dataclass
class SlateQInput(BaseInput):
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
class ParametricDqnInput(BaseInput):
    action: FeatureData
    next_action: FeatureData
    possible_actions: FeatureData
    possible_actions_mask: torch.Tensor
    possible_next_actions: FeatureData
    possible_next_actions_mask: torch.Tensor
    extras: Optional[ExtraData] = None
    weight: Optional[torch.Tensor] = None

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
            weight=batch.get("weight", None),
        )


@dataclass
class PolicyNetworkInput(BaseInput):
    action: FeatureData
    next_action: FeatureData
    extras: Optional[ExtraData] = None

    @classmethod
    def from_dict(cls, batch):
        base = super().from_dict(batch)
        # TODO: Implement ExtraData.from_dict
        extras = batch.get("extras", None)
        return cls(
            action=FeatureData(float_features=batch["action"]),
            next_action=FeatureData(float_features=batch["next_action"]),
            extras=extras,
            **base.as_dict_shallow(),
        )


@dataclass
class PolicyGradientInput(TensorDataClass):
    """
    See input_prototype for expected input dimensions
    """

    state: FeatureData
    action: torch.Tensor
    reward: torch.Tensor
    log_prob: torch.Tensor
    possible_actions_mask: Optional[torch.Tensor] = None

    @classmethod
    def input_prototype(cls, action_dim=2, batch_size=10, state_dim=3):
        return cls(
            state=FeatureData(float_features=torch.randn(batch_size, state_dim)),
            action=F.one_hot(
                torch.randint(high=action_dim, size=(batch_size,)),
                num_classes=action_dim,
            ),
            reward=torch.rand(batch_size),
            log_prob=torch.log(torch.rand(batch_size)),
            possible_actions_mask=torch.ones(batch_size, action_dim),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]):
        # TODO: rename "observation" to "state" in Transition and return cls(**d)
        return cls(
            state=FeatureData(float_features=d["observation"]),
            action=d["action"],
            reward=d["reward"],
            log_prob=d["log_prob"],
            possible_actions_mask=d.get("possible_actions_mask", None),
        )

    def __len__(self):
        assert self.action.ndim == 2
        return len(self.action)

    def batch_size(self):
        return len(self)


@dataclass
class BanditRewardModelInput(TensorDataClass):
    state: FeatureData
    action: torch.Tensor
    reward: torch.Tensor
    action_prob: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, batch: Dict[str, torch.Tensor]):
        return cls(
            state=FeatureData(float_features=batch["state_features"]),
            action=batch["action"],
            reward=batch["reward"],
            action_prob=batch.get("action_probability", None),
        )

    def batch_size(self):
        assert self.state.float_features.ndim == 2
        return self.state.float_features.size()[0]


@dataclass
class BehavioralCloningModelInput(TensorDataClass):
    state: FeatureData
    action: torch.Tensor
    possible_actions_mask: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, batch: Dict[str, torch.Tensor]):
        return cls(
            state=FeatureData(float_features=batch["state"]),
            action=batch["action"],
            possible_actions_mask=batch.get("possible_actions_mask", None),
        )

    def batch_size(self):
        assert self.state.float_features.ndim == 2
        return self.state.float_features.size()[0]


@dataclass
class MemoryNetworkInput(BaseInput):
    action: FeatureData
    valid_step: Optional[torch.Tensor] = None
    extras: ExtraData = field(default_factory=ExtraData)

    @classmethod
    def from_dict(cls, d):
        return cls(
            state=FeatureData(
                float_features=d["state"],
            ),
            next_state=FeatureData(
                float_features=d["next_state"],
            ),
            action=FeatureData(float_features=d["action"]),
            reward=d["reward"],
            time_diff=d["time_diff"],
            not_terminal=d["not_terminal"],
            step=d["step"],
            extras=ExtraData.from_dict(d),
        )

    def __len__(self):
        if len(self.state.float_features.size()) == 2:
            return self.state.float_features.size()[0]
        elif len(self.state.float_features.size()) == 3:
            return self.state.float_features.size()[1]
        else:
            raise NotImplementedError()


@dataclass
class SlateScoreBatch:
    mdp_id: torch.Tensor
    sequence_number: torch.Tensor
    scores: torch.Tensor
    training_input: PolicyGradientInput


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
class Seq2RewardOutput(TensorDataClass):
    acc_reward: torch.Tensor


@dataclass
class DqnPolicyActionSet(TensorDataClass):
    greedy: int
    softmax: Optional[int] = None
    greedy_act_name: Optional[str] = None
    softmax_act_name: Optional[str] = None
    softmax_act_prob: Optional[float] = None


@dataclass
class PlanningPolicyOutput(TensorDataClass):
    # best action to take next
    next_best_continuous_action: Optional[torch.Tensor] = None
    next_best_discrete_action_one_hot: Optional[torch.Tensor] = None
    next_best_discrete_action_idx: Optional[int] = None


@dataclass
class RankingOutput(TensorDataClass):
    # a tensor of integer indices w.r.t. to possible candidates
    # the values are offset by 2 to account for padding and decoder-starter symbol
    # shape: batch_size, tgt_seq_len
    # e.g., there are candidates C0, C1, C2, C3, C4, and the ranked order is
    # C4, C1, C2, C3, C0. Then the ranked_tgt_out_idx = [6, 3, 4, 5, 2]
    ranked_tgt_out_idx: Optional[torch.Tensor] = None

    # generative probability of ranked tgt sequences at each decoding step
    # shape: batch_size, tgt_seq_len, candidate_size
    ranked_per_symbol_probs: Optional[torch.Tensor] = None

    # generative probability of ranked tgt sequences
    # shape: batch_size, 1
    ranked_per_seq_probs: Optional[torch.Tensor] = None

    # log probabilities of given tgt sequences are used in REINFORCE
    # shape: batch_size, 1 if Seq2SlateMode == PER_SEQ_LOG_PROB_MODE
    # shape: batch_size, tgt_seq_len if Seq2SlateMode == PER_SYMBOL_LOG_PROB_DIST_MODE
    log_probs: Optional[torch.Tensor] = None
    # encoder scores in tgt_out_idx order
    encoder_scores: Optional[torch.Tensor] = None


@dataclass
class RewardNetworkOutput(TensorDataClass):
    predicted_reward: torch.Tensor


@dataclass
class SyntheticRewardNetworkOutput(TensorDataClass):
    predicted_reward: torch.Tensor
    mask: torch.Tensor
    output: torch.Tensor


@dataclass
class FrechetSortConfig:
    shape: float
    equiv_len: int
    topk: Optional[int] = None
    log_scores: bool = True


@dataclass
class CBInput(TensorDataClass):
    context_arm_features: torch.Tensor
    features_of_chosen_arm: Final[Optional[torch.Tensor]] = None
    arm_presence: Final[Optional[torch.Tensor]] = None
    action: Final[Optional[torch.Tensor]] = None  # chosen arm
    reward: Final[
        Optional[torch.Tensor]
    ] = None  # reward of the chosen arm. Used mostly for Offline Evalution. `label` is used as target for training.
    label: Final[
        Optional[torch.Tensor]
    ] = None  # label used for model training. Could be same as `reward`, or something else.
    rewards_all_arms: Final[
        Optional[torch.Tensor]
    ] = None  # rewards of all arms of the episode
    log_prob: Final[Optional[torch.Tensor]] = None
    weight: Final[Optional[torch.Tensor]] = None
    arms: Final[Optional[torch.Tensor]] = None
    mdp_id: Final[Optional[torch.Tensor]] = None

    @classmethod
    def input_prototype(
        cls,
        context_dim: int = 2,
        batch_size: int = 10,
        arm_features_dim: int = 3,
        num_arms: int = 4,
    ) -> "CBInput":
        return cls(
            context_arm_features=torch.randn(batch_size, num_arms, arm_features_dim)
        )

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "CBInput":
        return cls(
            context_arm_features=d["context_arm_features"],
            arm_presence=d.get("arm_presence", None),
            action=d.get("action", None),
            reward=d.get("reward", None),
            label=d.get("label", None),
            log_prob=d.get("log_prob", None),
            weight=d.get("weight", None),
            arms=d.get("arms", None),
            mdp_id=d.get("mdp_id", None),
        )

    def __len__(self) -> int:
        return self.context_arm_features.shape[0]

    @property
    def device(self) -> torch.device:
        return self.context_arm_features.device

    def __post_init__(self):
        if self.label is None and self.reward is not None:
            self.label = self.reward.clone()
