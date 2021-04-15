#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Callable, List, Optional

import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import NormalizationData
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.preprocessing.sparse_preprocessor import make_sparse_preprocessor


logger = logging.getLogger(__name__)


class Compose:
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
    """ For simple transforms """

    def __init__(self, keys: List[str], fn: Callable):
        self.keys = keys
        self.fn = fn

    def __call__(self, data):
        for k in self.keys:
            data[k] = self.fn(data[k])
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
            data[k] = self._preprocessor(value, presence)

        return data


class MapIDListFeatures:
    def __init__(
        self,
        id_list_keys: List[str],
        id_score_list_keys: List[str],
        feature_config: rlt.ModelFeatureConfig,
        device: torch.device,
    ):
        self.id_list_keys = id_list_keys
        self.id_score_list_keys = id_score_list_keys
        assert set(id_list_keys).intersection(set(id_score_list_keys)) == set()
        self.feature_config = feature_config
        self.sparse_preprocessor = make_sparse_preprocessor(
            feature_config=feature_config, device=device
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
    """Keys should be in the set {0,1,2,...,num_actions}, where
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


class MaskByPresence:
    """
    Expect data to be (value, presence) and return value * presence.
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
    Expect data to be List of (Value, Presence), and output a tensor of shape
    (batch_size, feature_dim).
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
    For loops over each key, to_key in zip(keys, to_keys).
    Expects each key to be `Dict[Int, Tuple[Tensor, T]]`.
    The sequence_id is the key of the dict. The first element of the tuple
    is the offset for each example, which is expected to be in fixed interval.
    If `to_key` is set, extract `T` to that key. Otherwise, put `T` back to `key`

    This is mainly for FB internal use,
    see fbcode/caffe2/caffe2/fb/proto/io_metadata.thrift
    for the data format extracted from SequenceFeatureMetadata

    NOTE: this is not product between two lists (keys and to_keys);
    it's setting keys[i] to to_keys[i] in a parallel way
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
            expected_length = self.expected_length
            if expected_length is None:
                if len(offsets) > 1:
                    # If batch size is larger than 1, just use the offsets
                    expected_length = (offsets[1] - offsets[0]).item()
                else:
                    # If batch size is 1
                    expected_length = value[0].shape[0]
                self.expected_length = expected_length
            expected_offsets = torch.arange(
                0, offsets.shape[0] * expected_length, expected_length
            )
            assert all(
                expected_offsets == offsets
            ), f"Unexpected offsets for {key} {self.sequence_id}: {offsets}. Expected {expected_offsets}"

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


class FixedLengthSequenceDenseNormalization:
    def __init__(
        self,
        keys: List[str],
        sequence_id: int,
        normalization_data: NormalizationData,
        expected_length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        to_keys = [f"{k}:{sequence_id}" for k in keys]
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
