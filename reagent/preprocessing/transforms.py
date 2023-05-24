#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import NormalizationData
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.preprocessing.sparse_preprocessor import make_sparse_preprocessor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger = logging.getLogger(__name__)


class Compose:
    """
    Applies an iterable collection of transform functions
    """

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        transforms = "\n    ".join([repr(t) for t in self.transforms])
        return f"{self.__class__.__name__}(\n{transforms}\n)"


# TODO: this wouldn't work for possible_actions_mask (list of value, presence)
class ValuePresence:
    """
    For every key `x`, looks for `x_presence`; if `x_presence` exists,
    replace `x` with tuple of `x` and `x_presence`, delete `x_presence` key
    """

    def __call__(self, data):
        keys = list(data.keys())

        for k in keys:
            presence_key = f"{k}_presence"
            if presence_key in data:
                data[k] = (data[k], data[presence_key])
                del data[presence_key]

        return data


class Lambda:
    """Applies an arbitrary callable transform"""

    def __init__(self, keys: List[str], fn: Callable):
        self.keys = keys
        self.fn = fn

    def __call__(self, data):
        for k in self.keys:
            data[k] = self.fn(data[k])
        return data


class SelectValuePresenceColumns:
    """
    Select columns from value-presence source key
    """

    def __init__(self, source: str, dest: str, indices: List[int]):
        self.source = source
        self.dest = dest
        self.indices = indices

    def __call__(self, data):
        value, presence = data[self.source]
        data[self.dest] = (value[:, self.indices], presence[:, self.indices])
        return data


class DenseNormalization:
    """
    Normalize the `keys` using `normalization_data`.
    The keys are expected to be `Tuple[torch.Tensor, torch.Tensor]`,
    where the first element is the value and the second element is the
    presence mask.
    This transform replaces the keys in the input data.
    """

    def __init__(
        self,
        keys: List[str],
        normalization_data: NormalizationData,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            keys: the name of the keys to be transformed
        """
        self.keys = keys
        self.normalization_data = normalization_data
        self.device = device or torch.device("cpu")
        # Delay the initialization of the preprocessor so this class
        # is pickleable
        self._preprocessor: Optional[Preprocessor] = None

    def __call__(self, data):
        if self._preprocessor is None:
            self._preprocessor = Preprocessor(
                self.normalization_data.dense_normalization_parameters,
                device=self.device,
            )

        for k in self.keys:
            value, presence = data[k]
            value, presence = value.to(self.device), presence.to(self.device)
            presence[torch.isnan(value)] = 0
            value[torch.isnan(value)] = 0
            data[k] = self._preprocessor(value, presence).float()

        return data


def _build_id_2_embedding_size(
    keys: List[str],
    feature_configs: List[List[rlt.BaseDataClass]],
    id_mapping_configs: List[Dict[str, rlt.IdMappingConfig]],
):
    """Sparse feature id -> embedding_table_size in corresponding id_mapping_config"""
    id_2_embedding_size = {}
    for key, feature_config, id_mapping_config in zip(
        keys, feature_configs, id_mapping_configs
    ):
        id_2_embedding_size[key] = {
            # pyre-fixme[16]: `BaseDataClass` has no attribute `feature_id`.
            config.feature_id: id_mapping_config[
                # pyre-fixme[16]: `BaseDataClass` has no attribute `id_mapping_name`.
                config.id_mapping_name
            ].embedding_table_size
            for config in feature_config
        }
    return id_2_embedding_size


def _build_id_2_hashing(
    keys: List[str],
    feature_configs: List[List[rlt.BaseDataClass]],
    id_mapping_configs: List[Dict[str, rlt.IdMappingConfig]],
):
    """Sparse feature id -> hashing boolean in corresponding id_mapping_config"""
    id_2_hashing = {}
    for key, feature_config, id_mapping_config in zip(
        keys, feature_configs, id_mapping_configs
    ):
        id_2_hashing[key] = {
            # pyre-fixme[16]: `BaseDataClass` has no attribute `feature_id`.
            # pyre-fixme[16]: `BaseDataClass` has no attribute `id_mapping_name`.
            config.feature_id: id_mapping_config[config.id_mapping_name].hashing
            for config in feature_config
        }
    return id_2_hashing


def _build_id_2_name(
    keys: List[str],
    feature_configs: List[List[rlt.BaseDataClass]],
):
    """Sparse feature id -> sparse feature name"""
    id_2_name = {}
    for key, feature_config in zip(keys, feature_configs):
        # pyre-fixme[16]: `BaseDataClass` has no attribute `feature_id`.
        # pyre-fixme[16]: `BaseDataClass` has no attribute `name`.
        id_2_name[key] = {config.feature_id: config.name for config in feature_config}
    return id_2_name


class IDListFeatures:
    """
    Process data read by SparseFeatureMetadata(sparse_feature_type=MULTI_CATEGORY) to KeyedJaggedTensor

    For source data format {key: (offsets, ids)}, see examples in fbcode/caffe2/caffe2/fb/proto/io_metadata.thrift:
    https://fburl.com/code/ndbg93s0

    For target data format, see examples in fbcode/torchrec/sparse/jagged_tensor.py:
    https://fburl.com/code/iad11zzc
    """

    def __init__(
        self,
        keys: List[str],
        feature_configs: List[List[rlt.IdListFeatureConfig]],
        id_mapping_configs: List[Dict[str, rlt.IdMappingConfig]],
    ):
        """
        Args:
            keys (List[str]): a list of columns to apply this transform
            feature_configs: a list of feature configs, corresponding to each column in keys
            id_mapping_configs: a list of id mapping configs, corresponding to each column in keys
        """
        self.keys = keys
        self.feature_configs = feature_configs
        self.id_mapping_configs = id_mapping_configs
        assert len(self.feature_configs) > 0, "No id list feature config provided"
        self._id_2_embed_size = _build_id_2_embedding_size(
            keys,
            # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but
            #  got `List[List[IdListFeatureConfig]]`.
            feature_configs,
            id_mapping_configs,
        )
        self._id_2_hashing = _build_id_2_hashing(
            keys,
            # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but
            #  got `List[List[IdListFeatureConfig]]`.
            feature_configs,
            id_mapping_configs,
        )
        # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but got
        #  `List[List[IdListFeatureConfig]]`.
        self._id_2_name = _build_id_2_name(keys, feature_configs)

    def __call__(self, data):
        for k in self.keys:
            jagged_tensor_keys: List[str] = []
            values: List[torch.Tensor] = []
            lengths: List[torch.Tensor] = []

            for feature_id in data[k].keys():
                feature_name = self._id_2_name[k][feature_id]
                jagged_tensor_keys.append(feature_name)
                offset, ids = data[k][feature_id]
                offset = torch.cat([offset, torch.tensor([len(ids)])])
                lengths.append(offset[1:] - offset[:-1])
                hashing = self._id_2_hashing[k][feature_id]
                if hashing:
                    embed_size = self._id_2_embed_size[k][feature_id]
                    hashed_ids = torch.ops.fb.sigrid_hash(
                        ids,
                        salt=0,
                        maxValue=embed_size,
                        hashIntoInt32=False,
                    )
                    values.append(hashed_ids)
                else:
                    values.append(ids)

            data[k] = KeyedJaggedTensor(
                keys=jagged_tensor_keys,
                values=torch.cat(values),
                lengths=torch.cat(lengths),
            )

        return data


class IDScoreListFeatures:
    """
    Process data read by SparseFeatureMetadata(sparse_feature_type=WEIGHTED_MULTI_CATEGORY) to KeyedJaggedTensor

    For source data format {key: (offsets, ids, weights)}, see examples in fbcode/caffe2/caffe2/fb/proto/io_metadata.thrift:
    https://fburl.com/code/ndbg93s0

    For target data format, see examples in fbcode/torchrec/sparse/jagged_tensor.py:
    https://fburl.com/code/iad11zzc
    """

    def __init__(
        self,
        keys: List[str],
        feature_configs: List[List[rlt.IdScoreListFeatureConfig]],
        id_mapping_configs: List[Dict[str, rlt.IdMappingConfig]],
    ):
        """
        Args:
            keys (List[str]): a list of columns to apply this transform
            feature_configs: a list of feature configs, corresponding to each column in keys
            id_mapping_configs: a list of id mapping configs, corresponding to each column in keys
        """
        self.keys = keys
        self.feature_configs = feature_configs
        self.id_mapping_configs = id_mapping_configs
        assert len(self.keys) == len(
            self.feature_configs
        ), "There should be as many keys as feature_configs"
        self._id_2_embed_size = _build_id_2_embedding_size(
            keys,
            # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but
            #  got `List[List[IdScoreListFeatureConfig]]`.
            feature_configs,
            id_mapping_configs,
        )
        self._id_2_hashing = _build_id_2_hashing(
            keys,
            # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but
            #  got `List[List[IdScoreListFeatureConfig]]`.
            feature_configs,
            id_mapping_configs,
        )
        # pyre-fixme[6]: For 2nd param expected `List[List[BaseDataClass]]` but got
        #  `List[List[IdScoreListFeatureConfig]]`.
        self._id_2_name = _build_id_2_name(keys, feature_configs)

    def __call__(self, data):
        for k in self.keys:
            jagged_tensor_keys: List[str] = []
            values: List[torch.Tensor] = []
            lengths: List[torch.Tensor] = []
            weights: List[torch.Tensor] = []

            for feature_id in data[k].keys():
                feature_name = self._id_2_name[k][feature_id]
                jagged_tensor_keys.append(feature_name)
                offset, ids, weight = data[k][feature_id]
                offset = torch.cat([offset, torch.tensor([len(ids)])])
                lengths.append(offset[1:] - offset[:-1])
                weights.append(weight)
                hashing = self._id_2_hashing[k][feature_id]
                if hashing:
                    embed_size = self._id_2_embed_size[k][feature_id]
                    hashed_ids = torch.ops.fb.sigrid_hash(
                        ids,
                        salt=0,
                        maxValue=embed_size,
                        hashIntoInt32=False,
                    )
                    values.append(hashed_ids)
                else:
                    values.append(ids)

            data[k] = KeyedJaggedTensor(
                keys=jagged_tensor_keys,
                values=torch.cat(values),
                lengths=torch.cat(lengths),
                weights=torch.cat(weights),
            )

        return data


class MapIDListFeatures:
    """
    Deprecated: Applies a SparsePreprocessor (see sparse_preprocessor.SparsePreprocessor)

    This class should be deprecated in favor of IDListFeatures and IDScoreListFeatures
    """

    def __init__(
        self,
        id_list_keys: List[str],
        id_score_list_keys: List[str],
        feature_config: rlt.ModelFeatureConfig,
        device: torch.device,
    ):
        self.id_list_keys = id_list_keys
        self.id_score_list_keys = id_score_list_keys
        assert (
            set(id_list_keys).intersection(set(id_score_list_keys)) == set()
        ), f"id_list_keys: {id_list_keys}; id_score_list_keys: {id_score_list_keys}"
        self.feature_config = feature_config
        self.sparse_preprocessor = make_sparse_preprocessor(
            feature_config=feature_config, device=device, gen_torch_script=False
        )

    def __call__(self, data):
        for k in self.id_list_keys + self.id_score_list_keys:
            # if no ids, it means we're not using sparse features.
            if not self.feature_config.id2name or k not in data:
                data[k] = None
                continue

            assert isinstance(data[k], dict), f"{k} has type {type(data[k])}. {data[k]}"
            if k in self.id_list_keys:
                data[k] = self.sparse_preprocessor.preprocess_id_list(data[k])
            else:
                data[k] = self.sparse_preprocessor.preprocess_id_score_list(data[k])
        return data


class OneHotActions:
    """
    Keys should be in the set {0,1,2,...,num_actions}, where
    a value equal to num_actions denotes that it's not valid.
    """

    def __init__(self, keys: List[str], num_actions: int):
        self.keys = keys
        self.num_actions = num_actions

    def __call__(self, data):
        for k in self.keys:
            # we do + 1 and then index up to n because value could be num_actions,
            # in which case the result is a zero-vector
            data[k] = F.one_hot(data[k], self.num_actions + 1).index_select(
                -1, torch.arange(self.num_actions)
            )
        return data


class ColumnVector:
    """
    Ensure that the keys are column vectors
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            raw_value = data[k]
            if isinstance(raw_value, tuple):
                value, _presence = raw_value
            elif isinstance(raw_value, list):
                # TODO(T67265031): make mdp_id a tensor, which we will be able to
                # when column type changes to int
                value = np.array(raw_value)
            elif isinstance(raw_value, torch.Tensor):
                # TODO(T67265031): this is an identity mapping, which is only necessary
                # when mdp_id in traced batch preprocessors becomes a tensor (mdp_id
                # is a list of strings in normal batch preprocessors).
                value = raw_value
            else:
                raise NotImplementedError(f"value of type {type(raw_value)}.")

            assert value.ndim == 1 or (
                value.ndim == 2 and value.shape[1] == 1
            ), f"Invalid shape for key {k}: {value.shape}"
            data[k] = value.reshape(-1, 1)

        return data


class ExtractValue:
    """
    Input is of format list(tuple(tensor, tensor)) - list(tuple(value, presence)).
    Output is of format list(tensor) with only the value extracted for the batch.

    Note that this transform works on array data type only.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data):
        extra_val_list = []
        for k in self.keys:
            raw_list = data[k]
            assert isinstance(
                raw_list, list
            ), f"Extra key - {k} must be of format list(tuple(tensor, tensor))"
            assert len(raw_list) != 0, f"Extra key - {k} cannot be an empty list"
            for raw_value in raw_list:
                value, presence = raw_value
                extra_val_list.append(value)
        data[k] = extra_val_list
        return data


class MaskByPresence:
    """
    Expect data to be (value, presence) and return value * presence.
    This zeros out values that aren't present.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            value_presence = data[k]
            assert (
                isinstance(value_presence, tuple) and len(value_presence) == 2
            ), f"Not valid value, presence tuple: {value_presence}"
            value, presence = value_presence
            assert value.shape == presence.shape, (
                f"Unmatching value shape ({value.shape})"
                f" and presence shape ({presence.shape})"
            )
            data[k] = value * presence.float()

        return data


class StackDenseFixedSizeArray:
    """
    If data is a tensor, ensures it has the correct shape. If data is a list of
    (value, presence) discards the presence tensors and concatenates the values
    to output a tensor of shape (batch_size, feature_dim).
    """

    def __init__(self, keys: List[str], size: int, dtype=torch.float):
        self.keys = keys
        self.size = size
        self.dtype = dtype

    def __call__(self, data):
        for k in self.keys:
            value = data[k]
            if isinstance(value, torch.Tensor):
                # Just ensure the shape
                if not (value.ndim == 2 and value.shape[1] == self.size):
                    raise ValueError(f"Wrong shape for key {k}: {value.shape}")
                data[k] = value.to(self.dtype)
            else:
                # Assuming that value is List[Tuple[torch.Tensor, torch.Tensor]]
                data[k] = (
                    torch.cat([v for v, p in value], dim=0)
                    .view(-1, self.size)
                    .to(dtype=self.dtype)
                )
        return data


class FixedLengthSequences:
    """
    Does two things:
        1. makes sure each sequence in the list of keys has the expected fixed length
        2. if to_keys is provided, copies the relevant sequence_id to the new key,
        otherwise overwrites the old key

    Expects each data[key] to be `Dict[Int, Tuple[Tensor, T]]`. Where:
    - key is the feature id
    - sequence_id is the key of the dict data[key]
    - The first element of the tuple is the offset for each example, which is expected to be in fixed interval.
    - The second element is the data at each step in the sequence

    This is mainly for FB internal use,
    see fbcode/caffe2/caffe2/fb/proto/io_metadata.thrift
    for the data format extracted from SequenceFeatureMetadata

    NOTE: this is not product between two lists (keys and to_keys);
    it's setting keys[sequence_id] to to_keys in a parallel way
    """

    def __init__(
        self,
        keys: List[str],
        sequence_id: int,
        expected_length: Optional[int] = None,
        *,
        to_keys: Optional[List[str]] = None,
    ):
        self.keys = keys
        self.sequence_id = sequence_id
        self.to_keys = to_keys or keys
        assert len(self.to_keys) == len(keys)
        self.expected_length = expected_length

    def __call__(self, data):
        for key, to_key in zip(self.keys, self.to_keys):
            offsets, value = data[key][self.sequence_id]
            # TODO assert regarding offsets length compared to value
            expected_length = self.expected_length
            if expected_length is None:
                if len(offsets) > 1:
                    # If batch size is larger than 1, just use the offsets
                    expected_length = (offsets[1] - offsets[0]).item()
                else:
                    # If batch size is 1
                    expected_length = value[0].size(0)
                self.expected_length = expected_length

            # some check that all arrays have the same length
            last_len = (value[0].size(0) - offsets[-1]).view(1)
            lengths = torch.cat((torch.diff(offsets), last_len))
            length = torch.unique(lengths)
            if not (len(length) == 1 and length == torch.tensor(self.expected_length)):
                raise ValueError(
                    f"Expected all batches for {key} to have {expected_length} items, but got sizes {lengths}"
                )

            data[to_key] = value
        return data


class SlateView:
    """
    Assuming that the keys are flatten fixed-length sequences with length of
    `slate_size`, unflatten it by inserting `slate_size` to the 1st dim.
    I.e., turns the input from the shape of `[B * slate_size, D]` to
    `[B, slate_size, D]`.
    """

    def __init__(self, keys: List[str], slate_size: int):
        self.keys = keys
        self.slate_size = slate_size

    def __call__(self, data):
        for k in self.keys:
            value = data[k]
            _, dim = value.shape
            data[k] = value.view(-1, self.slate_size, dim)

        return data


class VarLengthSequences:
    """
    Like FixedLengthSequences, but doesn't require the sequence-lengths to be the same. Instead,
        the largest slate size from the batch is used. For batches with smaller
        slate sizes, the values are padded with zeros.
    Additionally a presence tensor is produced to indicate which elements are present
        vs padded.
    The item presense tensor is a float boolean tensor of shape `[B, max_slate_size]`
    """

    def __init__(
        self,
        keys: List[str],
        sequence_id: int,
        *,
        to_keys: Optional[List[str]] = None,
        to_keys_item_presence: Optional[List[str]] = None,
    ):
        self.keys = keys
        self.sequence_id = sequence_id
        self.to_keys = to_keys or keys
        self.to_keys_item_presence = to_keys_item_presence or [
            k + "_item_presence" for k in self.to_keys
        ]
        assert len(self.to_keys) == len(keys)

    def __call__(self, data):
        for key, to_key, to_key_item_presence in zip(
            self.keys, self.to_keys, self.to_keys_item_presence
        ):
            # ignore the feature presence
            offsets, (value, presence) = data[key][self.sequence_id]

            # compute the length of each observation
            lengths = torch.diff(
                torch.cat(
                    (
                        offsets,
                        torch.tensor(
                            [value.shape[0]], dtype=offsets.dtype, device=offsets.device
                        ),
                    )
                )
            )

            num_obs = len(lengths)
            max_len = lengths.max().item()
            self.max_len = max_len
            feature_dim = value.shape[1]

            # create an empty 2d tensor to store the amended tensor
            # the new shape should be the maximum length of the observations times the number of observations, and the number of features
            new_shape = (num_obs * max_len, feature_dim)
            padded_value = torch.zeros(
                *new_shape, dtype=value.dtype, device=value.device
            )
            padded_presence = torch.zeros(
                *new_shape, dtype=presence.dtype, device=presence.device
            )

            # create a tensor of indices to scatter the values to
            indices = torch.cat(
                [
                    torch.arange(lengths[i], device=value.device) + i * max_len
                    for i in range(num_obs)
                ]
            )

            # insert the values into the padded tensor
            padded_value[indices] = value
            padded_presence[indices] = presence

            # get the item presence tensor
            item_presence = torch.cat(
                [
                    (torch.arange(max_len, device=value.device) < lengths[i]).float()
                    for i in range(num_obs)
                ]
            )

            item_presence = item_presence.view(-1, max_len)

            data[to_key] = (padded_value, padded_presence)
            data[to_key_item_presence] = item_presence

        return data


class FixedLengthSequenceDenseNormalization:
    """
    Combines the FixedLengthSequences, DenseNormalization, and SlateView transforms
    """

    def __init__(
        self,
        keys: List[str],
        sequence_id: int,
        normalization_data: NormalizationData,
        expected_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        to_keys: Optional[List[str]] = None,
    ):
        to_keys = to_keys or [f"{k}:{sequence_id}" for k in keys]
        self.fixed_length_sequences = FixedLengthSequences(
            keys, sequence_id, to_keys=to_keys, expected_length=expected_length
        )
        self.dense_normalization = DenseNormalization(
            to_keys, normalization_data, device=device
        )
        # We will override this in __call__()
        self.slate_view = SlateView(to_keys, slate_size=-1)

    def __call__(self, data):
        data = self.fixed_length_sequences(data)
        data = self.dense_normalization(data)
        self.slate_view.slate_size = self.fixed_length_sequences.expected_length
        return self.slate_view(data)


class VarLengthSequenceDenseNormalization:
    """
    Combines the VarLengthSequences, DenseNormalization, and SlateView transforms.
    For SlateView we infer the slate size at runtime and patch the transform.
    """

    def __init__(
        self,
        keys: List[str],
        sequence_id: int,
        normalization_data: NormalizationData,
        to_keys_item_presence: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        to_keys: Optional[List[str]] = None,
    ):
        to_keys = to_keys or [f"{k}:{sequence_id}" for k in keys]
        self.var_length_sequences = VarLengthSequences(
            keys,
            sequence_id,
            to_keys=to_keys,
            to_keys_item_presence=to_keys_item_presence,
        )
        self.dense_normalization = DenseNormalization(
            to_keys, normalization_data, device=device
        )
        # We will override slate_size in __call__()
        self.slate_view = SlateView(to_keys, slate_size=-1)

    def __call__(self, data):
        data = self.var_length_sequences(data)
        data = self.dense_normalization(data)
        self.slate_view.slate_size = (
            self.var_length_sequences.max_len
        )  # this assumes that max_len is the same for all all keys
        return self.slate_view(data)


class AppendConstant:
    """
    Append a column of constant value at the beginning of the specified dimension
    Can be used to add a column of "1" to the Linear Regression input data to capture intercept/bias
    """

    def __init__(self, keys: List[str], dim: int = -1, const: float = 1.0):
        self.keys = keys
        self.dim = dim
        self.const = const

    def __call__(self, data):
        for k in self.keys:
            value = data[k]
            extra_col = self.const * torch.ones(
                value.shape[:-1], device=value.device
            ).unsqueeze(-1)
            data[k] = torch.cat((extra_col, value), dim=self.dim)
        return data


class UnsqueezeRepeat:
    """
    This transform adds an extra dimension to the tensor and repeats
        the tensor along that dimension
    """

    def __init__(self, keys: List[str], dim: int, num_repeat: int = 1):
        self.keys = keys
        self.dim = dim
        self.num_repeat = num_repeat

    def __call__(self, data):
        for k in self.keys:
            data[k] = data[k].unsqueeze(self.dim)
            if self.num_repeat != 1:
                repeat_counters = [1 for _ in range(data[k].ndim)]
                repeat_counters[self.dim] = self.num_repeat
                data[k] = data[k].repeat(*repeat_counters)
        return data


def _get_product_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Get outer product of 2 tensors along the last dimension.
    All dimensions except last are preserved. The last dimension is replaced
        with flattened outer products of last-dimension-vectors from input tensors

    This is a vectorized implementation of (for 2D case):
    for i in range(x.shape[0]):
        out[i, :] = torch.outer(x[i, :], y[i, :]).flatten()

    For 2D inputs:
        Input shapes:
            x: (batch, feature_dim_x)
            y: (batch, feature_dim_y)
        Output shape:
            (batch, feature_dim_x*feature_dim_y)
    """
    return torch.einsum("...i,...j->...ij", (x, y)).flatten(start_dim=-2)


class OuterProduct:
    """
    This transform creates a tensor with an outer product of elements of 2 tensors.
    The outer product is stored under the new key.
    The 2 input tensors might be dropped, depending on input arguments
    """

    def __init__(
        self,
        key1: str,
        key2: str,
        output_key: str,
        drop_inputs: bool = False,
    ):
        self.key1 = key1
        self.key2 = key2
        self.output_key = output_key
        self.drop_inputs = drop_inputs

    def __call__(self, data):
        x = data[self.key1]
        y = data[self.key2]
        prod = _get_product_features(x, y)
        data[self.output_key] = prod
        if self.drop_inputs:
            del data[self.key1], data[self.key2]
        return data


class GetEye:
    """
    Place a diagonal tensor into the data dictionary
    """

    def __init__(self, key: str, size: int):
        self.key = key
        self.size = size

    def __call__(self, data):
        x = torch.eye(self.size)
        data[self.key] = x
        return data


def _broadcast_tensors_for_cat(
    tensors: List[torch.Tensor], dim: int
) -> List[torch.Tensor]:
    """
    Broadcast all tensors so that they could be concatenated along the specific dim.
    The tensor shapes have to be broadcastable (after the concatenation dim is taken out)

    Example:
    Input tensors of shapes [(10,3,5), (1,3,3)] (dim=2) would get broadcasted to [(10,3,5), (10,3,3)],
        so that they could be concatenated along the last dim.
    """
    if dim >= 0:
        dims = [dim] * len(tensors)
    else:
        dims = [t.ndim + dim for t in tensors]
    shapes = [list(t.shape) for t in tensors]
    for s, d in zip(shapes, dims):
        s.pop(d)
    shapes_except_cat_dim = [tuple(s) for s in shapes]
    broadcast_shape = torch.broadcast_shapes(*shapes_except_cat_dim)
    final_shapes = [list(broadcast_shape) for t in tensors]
    for s, t, d in zip(final_shapes, tensors, dims):
        s.insert(d, t.shape[dim])
    final_shapes = [tuple(s) for s in final_shapes]
    return [t.expand(s) for t, s in zip(tensors, final_shapes)]


class Cat:
    """
    This transform concatenates the tensors along a specified dim
    """

    def __init__(
        self, input_keys: List[str], output_key: str, dim: int, broadcast: bool = True
    ):
        self.input_keys = input_keys
        self.output_key = output_key
        self.dim = dim
        self.broadcast = broadcast

    def __call__(self, data):
        tensors = []
        for k in self.input_keys:
            tensors.append(data[k])
        if self.broadcast:
            tensors = _broadcast_tensors_for_cat(tensors, self.dim)
        data[self.output_key] = torch.cat(tensors, dim=self.dim)
        return data


class Rename:
    """
    Change key names
    """

    def __init__(self, old_names: List[str], new_names: List[str]):
        self.old_names = old_names
        self.new_names = new_names

    def __call__(self, data):
        new_data = dict(data)
        for o, n in zip(self.old_names, self.new_names):
            new_data[n] = new_data.pop(o)
        return new_data


class Filter:
    """
    Remove some keys from the dict.
    Can specify keep_keys (they will be kept) or remove_keys (they will be removed)
    """

    def __init__(
        self,
        *,
        keep_keys: Optional[List[str]] = None,
        remove_keys: Optional[List[str]] = None,
    ):
        assert (keep_keys is None) != (remove_keys is None)
        self.keep_keys = keep_keys
        self.remove_keys = remove_keys

    def __call__(self, data):
        if self.keep_keys:
            new_data = {}
            for k in self.keep_keys:
                if k in data:
                    new_data[k] = data[k]
        else:
            new_data = dict(data)
            for k in self.remove_keys:
                if k in new_data:
                    del new_data[k]
        return new_data


class ToDtype:
    """
    Convert tensors to a specific dtype
    """

    def __init__(self, dtypes: Dict[str, torch.dtype]):
        self.dtypes = dtypes

    def __call__(self, data):
        new_data = dict(data)
        for key, dtype in self.dtypes.items():
            new_data[key] = data[key].to(dtype=dtype)
        return new_data
