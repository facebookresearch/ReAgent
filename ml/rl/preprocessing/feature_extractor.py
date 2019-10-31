#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import dataclasses
from collections import OrderedDict
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import torch
from caffe2.python import core, schema
from ml.rl import types as rlt
from ml.rl.caffe_utils import C2
from ml.rl.preprocessing.normalization import (
    MISSING_VALUE,
    NormalizationParameters,
    get_feature_start_indices,
    get_num_output_features,
    sort_features_by_normalization,
)

from .preprocessor_net import PreprocessorNet


"""
These are set of classes facilitating interface between Caffe2-style data source
and PyTorch model
"""


class FeatureExtractorNet(NamedTuple):
    """
    `init_net` will only be run once. The external outputs of `init_net` are
    assumed to be parameters of `net` and will be saved in the predictor file.
    `net` should not use any parameters not initialized by `init_net`.
    """

    net: core.Net
    init_net: core.Net


class FeatureExtractorBase(object, metaclass=abc.ABCMeta):
    """
    This is not a transformer because Caffe2 has a weird API. We cannot directly
    call functions. It's easier to handle separately.
    """

    def __init__(self, model_feature_config: Optional[rlt.ModelFeatureConfig] = None):
        super().__init__()
        self._init_sequence_features(model_feature_config)

    def _init_sequence_features(self, config: Optional[rlt.ModelFeatureConfig]) -> None:
        self.id_mapping_configs = config.id_mapping_config if config else {}
        self.sequence_features_type = config.sequence_features_type if config else None
        sequence_features = (
            dataclasses.fields(self.sequence_features_type)
            if self.sequence_features_type
            else []
        )
        self.has_sequence_features = bool(sequence_features)
        self.sequence_features = OrderedDict(
            (s.name, s.type) for s in sequence_features
        )

        def get_id_features(t):
            fields = dataclasses.fields(t)
            for f in fields:
                # If the `id_features` remains an Optional, it won't be a type object
                if f.name != "id_features" or not isinstance(f.type, type):
                    continue
                return f.type.get_feature_config()
            return {}

        self.sequence_id_features = {
            s.name: get_id_features(s.type) for s in sequence_features
        }

        def get_id_feature_type(t):
            fields = dataclasses.fields(t)
            for f in fields:
                if f.name != "id_features":
                    continue
                return f.type
            return None

        self.sequence_id_feature_types = {
            k: get_id_feature_type(self.sequence_features[k])
            for k, v in self.sequence_id_features.items()
            if v
        }
        self.has_sequence_id_features = any(
            bool(v) for v in self.sequence_id_features.values()
        )
        self.has_sequence_float_features = any(
            bool(v.get_float_feature_infos()) for v in self.sequence_features.values()
        )

    def extract(self, ws, extract_record):
        """
        If the extractor is to be run, e.g., by the reader, then subclass should
        implement

        Args:
            extract_record (schema.Field): the output the net
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_net(self) -> FeatureExtractorNet:
        """
        Returns FeatureExtractorNet to perform feature extraction.

        The returned net must have input & output record set so that it can be
        bound to actual inputs/outputs
        """
        pass

    def create_const(  # type: ignore
        self, init_net, name, value, dtype=core.DataType.FLOAT  # type: ignore
    ) -> core.BlobReference:
        blob: core.BlobReference = init_net.NextScopedBlob(name)
        if not isinstance(value, list):
            shape: List[int] = []
            value = [value]
        else:
            shape = [len(value)]
        init_net.GivenTensorFill([], blob, shape=shape, values=value, dtype=dtype)
        init_net.AddExternalOutput(blob)
        return blob

    def extract_float_features(
        self, net, name, field, keys_to_extract, missing_scalar
    ) -> Tuple[str, str]:
        """
        Helper function to extract matrix from stacked  sparse tensors
        """
        with core.NameScope(name):
            float_features, presence_mask = net.SparseToDenseMask(
                [field.keys(), field.values(), missing_scalar, field.lengths()],
                [net.NextBlob("float_features"), net.NextBlob("presence_mask")],
                mask=keys_to_extract,
                return_presence_mask=True,
            )
        return (float_features, presence_mask)

    def get_state_sequence_features_schema(
        self,
        sequence_id_features: Dict[str, Dict[str, core.BlobReference]],
        sequence_float_features: Dict[str, core.BlobReference],
    ) -> schema.Struct:
        """
        Layout the record to match SequenceFeatures type. This is necessary to make ONNX
        exporting works.
        """
        record_fields = []
        for sequence_name, sequence_type in self.sequence_features.items():
            sequence_record = schema.Struct()
            if sequence_name in self.sequence_id_feature_types:
                fields = dataclasses.fields(
                    self.sequence_id_feature_types[sequence_name]
                )
                sequence_record += schema.Struct(
                    (
                        "id_features",
                        schema.Struct(
                            *[
                                (f.name, sequence_id_features[sequence_name][f.name])
                                for f in fields
                            ]
                        ),
                    )
                )

            if sequence_type.get_float_feature_infos():
                sequence_record += schema.Struct(
                    ("float_features", sequence_float_features[sequence_name])
                )
            record_fields.append((sequence_name, sequence_record))

        return schema.Struct(*record_fields)

    def extract_sequence_id_features(
        self,
        net: core.Net,
        name: str,
        sequence_feature_types: Dict[str, Type[rlt.SequenceFeatureBase]],
        sequence_id_features: Dict[str, Dict[str, rlt.IdFeatureConfig]],
        field: schema.List,
        empty_range: schema.BlobReference,
        zero_int64: schema.BlobReference,
    ) -> Dict[str, Dict[str, core.BlobReference]]:
        """
        Convert CSR-like format of MAP<BIGINT, LIST<BIGINT>> to dictionary from
        sequence name to dictionary from ID-list name to the blob containing the
        fixed-length sequence of IDs. Each blob will be 2-D tensor. The first dimension
        is the batch size. The second dimension is each element in the list.
        """
        feature_names: List[str] = []
        feature_ids: List[int] = []
        for sequence_name, id_features in sequence_id_features.items():
            for id_feature_name, id_feature_config in id_features.items():
                feature_names.append("{}_{}".format(sequence_name, id_feature_name))
                feature_ids.append(id_feature_config.feature_id)

        id_list_feature_ranges = self.extract_id_list_features_ranges(
            net, name, field, feature_names, feature_ids, empty_range
        )

        with core.NameScope(name):
            return {
                sequence_name: {
                    id_feature_name: self.range_to_dense(
                        net,
                        "{}_{}".format(sequence_name, id_feature_name),
                        id_list_feature_ranges[
                            "{}_{}".format(sequence_name, id_feature_name)
                        ]["ranges"],
                        id_list_feature_ranges[
                            "{}_{}".format(sequence_name, id_feature_name)
                        ]["values"],
                        sequence_feature_types[sequence_name].get_max_length(),
                        zero_int64,
                    )
                    for id_feature_name, id_feature_config in id_features.items()
                }
                for sequence_name, id_features in sequence_id_features.items()
                if id_features
            }

    def extract_sequence_float_features(
        self,
        net: core.Net,
        name: str,
        sequence_feature_types: Dict[str, Type[rlt.SequenceFeatureBase]],
        field: schema.List,
        empty_range: schema.BlobReference,
        zero_float: schema.BlobReference,
    ) -> Dict[str, core.BlobReference]:
        """
        Convert CSR-like format of MAP<BIGINT, MAP<BIGINT, FLOAT>> to dictionary from
        sequence name to the blob containing the fixed-length sequence of vector of
        float features of each element. Each blob will be 3-D tensor. The first
        dimension is the batch size. The second dimension is each element in the list.
        The third dimension is ordered by the order given by
        `SequenceFeatureBase.get_float_feature_infos()`. These float features are not
        normalized.
        """
        feature_names: List[str] = []
        feature_ids: List[int] = []

        for sequence_name, sequence_type in sequence_feature_types.items():
            for info in sequence_type.get_float_feature_infos():
                feature_names.append("{}_{}".format(sequence_name, info.name))
                feature_ids.append(info.feature_id)

        id_score_list_feature_ranges = self.extract_id_score_list_features_ranges(
            net, name, field, feature_names, feature_ids, empty_range
        )

        with core.NameScope(name):
            return {
                sequence_name: net.Concat(
                    [
                        self.range_to_dense(
                            net,
                            "{}_{}".format(sequence_name, info.name),
                            id_score_list_feature_ranges[
                                "{}_{}".format(sequence_name, info.name)
                            ]["ranges"],
                            id_score_list_feature_ranges[
                                "{}_{}".format(sequence_name, info.name)
                            ]["scores"],
                            sequence_type.get_max_length(),
                            zero_float,
                        )
                        for info in sequence_type.get_float_feature_infos()
                    ],
                    [sequence_name, "{}_split_info".format(sequence_name)],
                    axis=2,
                    add_axis=1,
                )[0]
                for sequence_name, sequence_type in sequence_feature_types.items()
                if sequence_type.get_float_feature_infos()
            }

    def create_empty_range(self, init_net: core.Net) -> core.BlobReference:
        return self.create_const(
            init_net, "empty_range", [0, 0], dtype=core.DataType.INT32  # type: ignore
        )

    def extract_id_list_features_ranges(
        self,
        net: core.Net,
        name: str,
        field: schema.List,
        feature_names: List[str],
        feature_ids: List[int],
        empty_range: core.BlobReference,
    ) -> Dict[str, Dict[str, core.BlobReference]]:
        """
        Convert the CSR-like format of ID-list to ranges and values.
        See https://caffe2.ai/docs/operators-catalogue#gatherranges

        The return value is keyed by ID-list name
        """
        assert len(feature_names) == len(
            feature_ids
        ), "feature_names and feature_ids must be parallel"
        with core.NameScope("{}_id_list_ranges".format(name)):
            id_list_ranges = net.LengthsToRanges(
                field["values"]["values"]["lengths"](), ["input_ranges"]
            )
            densified_ranges = net.SparseToDenseMask(
                [
                    field["values"]["keys"](),
                    id_list_ranges,
                    empty_range,
                    field["lengths"](),
                ],
                ["densified_ranges"],
                mask=feature_ids,
            )
            result = {}
            for idx, name in enumerate(feature_names):
                starts = [0, idx, 0]
                ends = [-1, idx + 1, -1]
                result[name] = {
                    "ranges": net.Slice(
                        [densified_ranges],
                        "{}_ranges".format(name),
                        starts=starts,
                        ends=ends,
                    ),
                    "values": field["values"]["values"]["values"](),
                }
        return result

    def extract_id_score_list_features_ranges(
        self,
        net: core.Net,
        name: str,
        field: schema.List,
        feature_names: List[str],
        feature_ids: List[int],
        empty_range: core.BlobReference,
    ) -> Dict[str, Dict[str, core.BlobReference]]:
        """
        Convert the CSR-like format of ID-score-list to ranges and values.
        See https://caffe2.ai/docs/operators-catalogue#gatherranges

        The return value is keyed by ID-score-list name
        """
        assert len(feature_names) == len(
            feature_ids
        ), "feature_names and feature_ids must be parallel"
        with core.NameScope("{}_id_score_list_ranges".format(name)):
            id_score_list_ranges = net.LengthsToRanges(
                field["values"]["values"]["lengths"](), ["input_ranges"]
            )

            densified_ranges = net.SparseToDenseMask(
                [
                    field["values"]["keys"](),
                    id_score_list_ranges,
                    empty_range,
                    field["lengths"](),
                ],
                ["densified_ranges"],
                mask=feature_ids,
            )
            result = {}
            for idx, name in enumerate(feature_names):
                starts = [0, idx, 0]
                ends = [-1, idx + 1, -1]
                result[name] = {
                    "ranges": net.Slice(
                        [densified_ranges],
                        "{}_ranges".format(name),
                        starts=starts,
                        ends=ends,
                    ),
                    "ids": field["values"]["values"]["values"]["keys"](),
                    "scores": field["values"]["values"]["values"]["values"](),
                }
        return result

    def range_to_dense(
        self,
        net: core.Net,
        name: str,
        ranges: core.BlobReference,
        values: core.BlobReference,
        max_length: int,
        zero_val: core.BlobReference,
    ) -> core.BlobReference:
        """
        Convert batch of variable-length lists (in range format) to fixed-length lists.
        """
        with core.NameScope("range_to_dense_{}".format(name)):
            # First slicing the offset and length
            offset = net.Cast(
                net.Slice(ranges, ["offset"], starts=[0, 0, 0], ends=[-1, -1, 1]),
                ["float_offset"],
                to=core.DataType.FLOAT,  # type: ignore
            )
            length = net.Cast(
                net.Slice(ranges, ["length"], starts=[0, 0, 1], ends=[-1, -1, 2]),
                ["float_length"],
                to=core.DataType.FLOAT,  # type: ignore
            )

            zero_offset = net.ConstantFill(length, ["zero_offset"], value=0.0)
            max_length_blob = net.ConstantFill(
                length, ["max_length"], value=float(max_length)
            )

            # Calculate the new offset, which is
            # offset + max(0, length - max_length)
            new_offset = net.Cast(
                net.Add(
                    [
                        offset,
                        net.Max(
                            [zero_offset, net.Sub([length, max_length_blob], ["sub"])],
                            ["max"],
                        ),
                    ],
                    ["float_new_offset"],
                    broadcast=1,
                ),
                ["new_offset"],
                to=core.DataType.INT32,  # type: ignore
            )
            new_length = net.Cast(
                net.Min([length, max_length_blob], "float_new_length"),
                ["new_length"],
                to=core.DataType.INT32,  # type: ignore
            )
            # Stitch these back together
            new_range, _ = net.Concat(
                [new_offset, new_length], ["new_range", "split_info"], axis=2
            )

            # At this point, we have lists w/ length up to max_length
            gathered_values, gathered_lengths = net.GatherRanges(
                [values, new_range], ["gathered_values", "gathered_length"]
            )
            # This generate indices for each element
            lengths_range_fill = net.LengthsRangeFill(
                gathered_lengths, ["lengths_range_fill"]
            )
            # Finally, we make the dense output
            keys_to_extract = list(range(max_length))
            values_with_mask: Tuple[
                core.BlobReference, core.BlobReference
            ] = net.SparseToDenseMask(
                [lengths_range_fill, gathered_values, zero_val, gathered_lengths],
                ["dense_values", "presence_mask"],
                mask=keys_to_extract,
                return_presence_mask=True,
            )
            dense_values, _presence_mask = values_with_mask

        return dense_values

    def create_id_mapping(
        self, init_net: core.Net, name: str, mapping: List[int]
    ) -> core.BlobReference:
        """
        Given the ID list in the mapping, create index from ID to its (1-base) index.
        """
        assert len(set(mapping)) == len(
            mapping
        ), "mapping for {} must not contain duplicated IDs".format(name)
        mapping_data = init_net.NextScopedBlob("mapping_data_{}".format(name))
        init_net.GivenTensorFill(
            [],
            mapping_data,
            shape=[len(mapping)],
            values=mapping,
            dtype=core.DataType.INT64,  # type: ignore
        )
        handler: core.BlobReference = init_net.NextScopedBlob("mapping_{}".format(name))
        init_net.LongIndexCreate([], handler, max_elements=len(mapping))
        init_net.IndexLoad([handler, mapping_data], [handler])
        init_net.IndexFreeze(handler, handler)
        init_net.AddExternalOutput(handler)
        return handler

    def create_id_mappings(
        self, init_net: core.Net, id_mapping_configs: Dict[str, rlt.IdMapping]
    ) -> Dict[str, core.BlobReference]:
        return {
            mapping_name: self.create_id_mapping(init_net, mapping_name, mapping.ids)
            for mapping_name, mapping in id_mapping_configs.items()
        }

    def map_ids(
        self,
        net: core.Net,
        name: str,
        map_handler: core.BlobReference,
        raw_ids: core.BlobReference,
    ) -> core.BlobReference:
        """
        Map raw ID to index (into embedding lookup table, usually)
        """
        with core.NameScope("mapping_{}".format(name)):
            retval: core.BlobReference = net.IndexGet(
                [map_handler, raw_ids], ["mapped_ids"]
            )
            assert type(retval) == core.BlobReference
            return retval

    def map_sequence_id_features(
        self,
        net: core.Net,
        name: str,
        map_handlers: Dict[str, core.BlobReference],
        raw_sequence_id_features: Dict[str, Dict[str, core.BlobReference]],
        sequence_id_feature_configs: Dict[str, Dict[str, rlt.IdFeatureConfig]],
    ) -> Dict[str, Dict[str, core.BlobReference]]:
        """
        Map raw IDs of all sequences' ID features to index (into embedding lookup table)
        """

        def _map_id_feature(sequence_name, id_feature, raw_id_feature):
            with core.NameScope(sequence_name):
                return self.map_ids(
                    net,
                    id_feature,
                    map_handlers[
                        sequence_id_feature_configs[sequence_name][
                            id_feature
                        ].id_mapping_name
                    ],
                    raw_id_feature,
                )

        with core.NameScope(name):
            return {
                sequence_name: {
                    id_feature: _map_id_feature(
                        sequence_name, id_feature, raw_id_feature
                    )
                    for id_feature, raw_id_feature in id_features.items()
                }
                for sequence_name, id_features in raw_sequence_id_features.items()
            }

    def fetch_state_sequence_features(
        self, record: schema.Struct, fetch_func
    ) -> rlt.SequenceFeatures:
        """
        Pull the values from Caffe2's blobs into PyTorch's tensors.
        """
        state_sequence_features = {}
        for seq_name, sequence_feature_type in self.sequence_features.items():
            state_seq = sequence_feature_type(id_features=None, float_features=None)

            if seq_name in self.sequence_id_feature_types:
                state_seq.id_features = self.sequence_id_feature_types[seq_name](
                    **{
                        feature_name: fetch_func(
                            record[seq_name]["id_features"][feature_name]
                        )
                        for feature_name in self.sequence_id_features[seq_name]
                    }
                )

            if sequence_feature_type.get_float_feature_infos():
                state_seq.float_features = fetch_func(
                    record[seq_name]["float_features"]
                )

            state_sequence_features[seq_name] = state_seq

        return self.sequence_features_type(**state_sequence_features)  # type: ignore

    def read_actions_to_mask(
        self, net, name, num_actions, action, action_size_plus_one
    ):
        with core.NameScope(name):
            action_blob_one_hot = net.OneHot(
                [action(), action_size_plus_one], ["action_blob_one_hot"]
            )
            action_blob_one_hot_sliced = net.Slice(
                [action_blob_one_hot],
                ["action_blob_one_hot_sliced"],
                starts=[0, 0],
                ends=[-1, num_actions],
            )
        return action_blob_one_hot_sliced

    @staticmethod
    def fetch(ws, b, to_torch=True):
        data = ws.fetch_blob(str(b()))
        if not isinstance(data, np.ndarray):
            # Blob uninitialized, return None and handle downstream
            assert False, "Missing blob: " + str(b)
            return None
        if to_torch:
            return torch.tensor(data)
        return data


def map_schema():
    return schema.Map(schema.Scalar(), schema.Scalar())


def id_list_schema():
    return schema.Map(schema.Scalar(), schema.List(schema.Scalar()))


def id_score_list_schema():
    return schema.Map(schema.Scalar(), schema.Map(schema.Scalar(), schema.Scalar()))


class InputColumn(object):
    STATE_FEATURES = "state_features"
    STATE_FEATURES_PRESENCE = "state_features_presence"
    STATE_ID_LIST_FEATURES = "state_id_list_features"
    STATE_ID_SCORE_LIST_FEATURES = "state_id_score_list_features"
    NEXT_STATE_FEATURES = "next_state_features"
    NEXT_STATE_FEATURES_PRESENCE = "next_state_features_presence"
    NEXT_STATE_ID_LIST_FEATURES = "next_state_id_list_features"
    NEXT_STATE_ID_SCORE_LIST_FEATURES = "next_state_id_score_list_features"
    ACTION = "action"
    ACTION_PRESENCE = "action_presence"
    NEXT_ACTION = "next_action"
    NEXT_ACTION_PRESENCE = "next_action_presence"
    POSSIBLE_ACTIONS = "possible_actions"
    POSSIBLE_ACTIONS_PRESENCE = "possible_actions_presence"
    POSSIBLE_ACTIONS_MASK = "possible_actions_mask"
    POSSIBLE_NEXT_ACTIONS = "possible_next_actions"
    POSSIBLE_NEXT_ACTIONS_PRESENCE = "possible_next_actions_presence"
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


class TrainingFeatureExtractor(FeatureExtractorBase):
    """
    Extract:
    - State
    - Action
    - Next state
    - Possible next actions/Next actions
    """

    def __init__(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None,
        include_possible_actions: bool = True,
        normalize: bool = True,
        max_num_actions: int = None,
        set_missing_value_to_zero: bool = None,
        multi_steps: Optional[int] = None,
        metrics_to_score: Optional[List[str]] = None,
        model_feature_config: Optional[rlt.ModelFeatureConfig] = None,
        use_time_since_first: Optional[bool] = None,
        time_since_first_normalization_parameters: Optional[
            NormalizationParameters
        ] = None,
    ) -> None:
        super().__init__(model_feature_config=model_feature_config)
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.sorted_state_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        if action_normalization_parameters:
            self.sorted_action_features, _ = sort_features_by_normalization(
                action_normalization_parameters
            )
        else:
            self.sorted_action_features = None
        self.include_possible_actions = include_possible_actions
        if action_normalization_parameters is None:
            self.include_possible_actions = True  # TODO: Delete SARSA in a future diff
        self.normalize = normalize
        self.max_num_actions = max_num_actions
        self.set_missing_value_to_zero = set_missing_value_to_zero
        self.multi_steps = multi_steps
        self.metrics_to_score = metrics_to_score
        self.use_time_since_first = use_time_since_first or False
        self.time_since_first_normalization_parameters = (
            time_since_first_normalization_parameters
        )

    def extract(self, ws, extract_record):
        fetch = partial(self.fetch, ws)

        def fetch_possible_actions(
            possible_actions_blob, possible_actions_presence_blob
        ):
            return rlt.FeatureVector(
                float_features=rlt.ValuePresence(
                    value=fetch(possible_actions_blob),
                    presence=fetch(possible_actions_presence_blob),
                )
            )

        state = rlt.FeatureVector(
            float_features=rlt.ValuePresence(
                value=fetch(extract_record.state_features),
                presence=fetch(extract_record.state_features_presence),
            )
        )

        next_state = rlt.FeatureVector(
            float_features=rlt.ValuePresence(
                value=fetch(extract_record.next_state_features),
                presence=fetch(extract_record.next_state_features_presence),
            )
        )

        if self.has_sequence_features:
            state = state._replace(
                sequence_features=self.fetch_state_sequence_features(
                    extract_record.state_sequence_features, fetch
                )
            )
            next_state = next_state._replace(
                sequence_features=self.fetch_state_sequence_features(
                    extract_record.next_state_sequence_features, fetch
                )
            )
        if self.use_time_since_first:
            state = state._replace(
                time_since_first=fetch(extract_record.time_since_first)
            )
            next_state = next_state._replace(
                time_since_first=fetch(extract_record.next_time_since_first)
            )

        if self.sorted_action_features is None:
            # Action is a one-hot vector of discrete actions
            action = fetch(extract_record.action)
            next_action = fetch(extract_record.next_action)
        else:
            # Action is a set of continuous features
            action = rlt.FeatureVector(
                float_features=rlt.ValuePresence(
                    value=fetch(extract_record.action),
                    presence=fetch(extract_record.action_presence),
                )
            )
            next_action = rlt.FeatureVector(
                float_features=rlt.ValuePresence(
                    value=fetch(extract_record.next_action),
                    presence=fetch(extract_record.next_action_presence),
                )
            )
        max_num_actions = None
        step = None
        if self.multi_steps is not None:
            step = fetch(extract_record.step).reshape(-1, 1)
        reward = fetch(extract_record.reward).reshape(-1, 1)

        # is_terminal should be filled by preprocessor
        not_terminal = fetch(extract_record.not_terminal).reshape(-1, 1)
        time_diff = fetch(extract_record.time_diff).reshape(-1, 1)

        if self.sorted_action_features is not None and self.max_num_actions is not None:
            tiled_next_state = rlt.FeatureVector(
                float_features=rlt.ValuePresence(
                    value=next_state.float_features.value.repeat(
                        1, self.max_num_actions
                    ).reshape(-1, next_state.float_features.value.shape[1]),
                    presence=next_state.float_features.presence.repeat(
                        1, self.max_num_actions
                    ).reshape(-1, next_state.float_features.value.shape[1]),
                )
            )
        else:
            tiled_next_state = None

        if self.include_possible_actions:
            # TODO: this will need to be more complicated to support sparse features
            assert self.max_num_actions is not None, "Missing max_num_actions"
            possible_actions_mask = (
                fetch(extract_record.possible_actions_mask)
                .reshape(-1, self.max_num_actions)
                .type(torch.FloatTensor)
            )
            possible_next_actions_mask = (
                fetch(extract_record.possible_next_actions_mask)
                .reshape(-1, self.max_num_actions)
                .type(torch.FloatTensor)
            )

            if self.sorted_action_features is not None:
                possible_actions = fetch_possible_actions(
                    extract_record.possible_actions,
                    extract_record.possible_actions_presence,
                )
                possible_next_actions = fetch_possible_actions(
                    extract_record.possible_next_actions,
                    extract_record.possible_next_actions_presence,
                )
                max_num_actions = self.max_num_actions

                training_input = rlt.RawParametricDqnInput(
                    state=state,
                    reward=reward,
                    time_diff=time_diff,
                    action=action,
                    next_action=next_action,
                    not_terminal=not_terminal,
                    next_state=next_state,
                    tiled_next_state=tiled_next_state,
                    possible_actions=possible_actions,
                    possible_actions_mask=possible_actions_mask,
                    possible_next_actions=possible_next_actions,
                    possible_next_actions_mask=possible_next_actions_mask,
                    step=step,
                )
            else:
                training_input = rlt.RawDiscreteDqnInput(
                    state=state,
                    reward=reward,
                    time_diff=time_diff,
                    action=action,
                    next_action=next_action,
                    not_terminal=not_terminal,
                    next_state=next_state,
                    possible_actions_mask=possible_actions_mask,
                    possible_next_actions_mask=possible_next_actions_mask,
                    step=step,
                )
        else:
            training_input = rlt.RawPolicyNetworkInput(
                state=state,
                reward=reward,
                time_diff=time_diff,
                action=action,
                next_action=next_action,
                not_terminal=not_terminal,
                next_state=next_state,
                step=step,
            )

        mdp_id = fetch(extract_record.mdp_id, to_torch=False)
        sequence_number = fetch(extract_record.sequence_number)

        metrics = fetch(extract_record.metrics) if self.metrics_to_score else None

        # TODO: stuff other fields in here
        extras = rlt.ExtraData(
            action_probability=fetch(extract_record.action_probability).reshape(-1, 1),
            sequence_number=sequence_number.reshape(-1, 1)
            if sequence_number is not None
            else None,
            mdp_id=mdp_id.reshape(-1, 1) if mdp_id is not None else None,
            max_num_actions=max_num_actions,
            metrics=metrics,
        )

        return rlt.RawTrainingBatch(training_input=training_input, extras=extras)

    def create_net(self):
        net = core.Net("feature_extractor")
        init_net = core.Net("feature_extractor_init")
        missing_scalar = self.create_const(
            init_net,
            "MISSING_SCALAR",
            0.0 if self.set_missing_value_to_zero else MISSING_VALUE,
        )

        action_schema = map_schema() if self.sorted_action_features else schema.Scalar()

        pass_through_columns = [
            (InputColumn.REWARD, schema.Scalar()),
            (InputColumn.NOT_TERMINAL, schema.Scalar()),
            (InputColumn.TIME_DIFF, schema.Scalar()),
            (InputColumn.MDP_ID, schema.Scalar()),
            (InputColumn.SEQUENCE_NUMBER, schema.Scalar()),
            (InputColumn.ACTION_PROBABILITY, schema.Scalar()),
        ]
        if self.multi_steps is not None:
            pass_through_columns.append((InputColumn.STEP, schema.Scalar()))

        input_schema = schema.Struct(
            *(
                [
                    (InputColumn.STATE_FEATURES, map_schema()),
                    (InputColumn.NEXT_STATE_FEATURES, map_schema()),
                    (InputColumn.ACTION, action_schema),
                    (InputColumn.NEXT_ACTION, action_schema),
                ]
                + pass_through_columns
            )
        )
        if self.has_sequence_id_features:
            input_schema += schema.Struct(
                (InputColumn.STATE_ID_LIST_FEATURES, id_list_schema()),
                (InputColumn.NEXT_STATE_ID_LIST_FEATURES, id_list_schema()),
            )
        if self.has_sequence_float_features:
            input_schema += schema.Struct(
                (InputColumn.STATE_ID_SCORE_LIST_FEATURES, id_score_list_schema()),
                (InputColumn.NEXT_STATE_ID_SCORE_LIST_FEATURES, id_score_list_schema()),
            )
        if self.include_possible_actions:
            input_schema += schema.Struct(
                (InputColumn.POSSIBLE_ACTIONS_MASK, schema.List(schema.Scalar())),
                (InputColumn.POSSIBLE_NEXT_ACTIONS_MASK, schema.List(schema.Scalar())),
            )
            if self.sorted_action_features is not None:
                input_schema += schema.Struct(
                    (InputColumn.POSSIBLE_ACTIONS, schema.List(map_schema())),
                    (InputColumn.POSSIBLE_NEXT_ACTIONS, schema.List(map_schema())),
                )

        if self.metrics_to_score:
            input_schema += schema.Struct((InputColumn.METRICS, map_schema()))

        if self.use_time_since_first:
            input_schema += schema.Struct(
                (InputColumn.TIME_SINCE_FIRST, schema.Scalar())
            )

        input_record = net.set_input_record(input_schema)

        state, state_presence = self.extract_float_features(
            net,
            "state",
            input_record[InputColumn.STATE_FEATURES],
            self.sorted_state_features,
            missing_scalar,
        )
        next_state, next_state_presence = self.extract_float_features(
            net,
            "next_state",
            input_record[InputColumn.NEXT_STATE_FEATURES],
            self.sorted_state_features,
            missing_scalar,
        )

        if self.has_sequence_features:
            empty_range = self.create_empty_range(init_net)

        if self.has_sequence_id_features:
            zero_int64 = self.create_const(
                init_net, "zero_int64", 0, dtype=core.DataType.INT64
            )
            state_sequence_id_features = self.extract_sequence_id_features(
                net,
                "state",
                self.sequence_features,
                self.sequence_id_features,
                input_record[InputColumn.STATE_ID_LIST_FEATURES],
                empty_range,
                zero_int64,
            )
            next_state_sequence_id_features = self.extract_sequence_id_features(
                net,
                "next_state",
                self.sequence_features,
                self.sequence_id_features,
                input_record[InputColumn.NEXT_STATE_ID_LIST_FEATURES],
                empty_range,
                zero_int64,
            )

            id_mappings = self.create_id_mappings(init_net, self.id_mapping_configs)
            state_sequence_id_features = self.map_sequence_id_features(
                net,
                "state",
                id_mappings,
                state_sequence_id_features,
                self.sequence_id_features,
            )
            next_state_sequence_id_features = self.map_sequence_id_features(
                net,
                "next_state",
                id_mappings,
                next_state_sequence_id_features,
                self.sequence_id_features,
            )
        else:
            state_sequence_id_features = {}
            next_state_sequence_id_features = {}

        if self.has_sequence_float_features:
            zero_float = self.create_const(init_net, "zero_float", 0.0)
            state_sequence_float_features = self.extract_sequence_float_features(
                net,
                "state",
                self.sequence_features,
                input_record[InputColumn.STATE_ID_SCORE_LIST_FEATURES],
                empty_range,
                zero_float,
            )
            next_state_sequence_float_features = self.extract_sequence_float_features(
                net,
                "next_state",
                self.sequence_features,
                input_record[InputColumn.NEXT_STATE_ID_SCORE_LIST_FEATURES],
                empty_range,
                zero_float,
            )
        else:
            state_sequence_float_features = {}
            next_state_sequence_float_features = {}

        if self.sorted_action_features:
            action, action_presence = self.extract_float_features(
                net,
                InputColumn.ACTION,
                input_record[InputColumn.ACTION],
                self.sorted_action_features,
                missing_scalar,
            )
            next_action, next_action_presence = self.extract_float_features(
                net,
                InputColumn.NEXT_ACTION,
                input_record[InputColumn.NEXT_ACTION],
                self.sorted_action_features,
                missing_scalar,
            )
            if self.include_possible_actions:
                possible_action_features, possible_action_features_presence = self.extract_float_features(
                    net,
                    InputColumn.POSSIBLE_ACTIONS,
                    input_record[InputColumn.POSSIBLE_ACTIONS]["values"],
                    self.sorted_action_features,
                    missing_scalar,
                )
                possible_next_action_features, possible_next_action_features_presence = self.extract_float_features(
                    net,
                    InputColumn.POSSIBLE_NEXT_ACTIONS,
                    input_record[InputColumn.POSSIBLE_NEXT_ACTIONS]["values"],
                    self.sorted_action_features,
                    missing_scalar,
                )
        else:
            action_size_plus_one = self.create_const(
                init_net,
                "action_size_plus_one",
                self.max_num_actions + 1,
                dtype=core.DataType.INT64,
            )
            action = self.read_actions_to_mask(
                net,
                InputColumn.ACTION,
                self.max_num_actions,
                input_record[InputColumn.ACTION],
                action_size_plus_one,
            )
            next_action = self.read_actions_to_mask(
                net,
                InputColumn.NEXT_ACTION,
                self.max_num_actions,
                input_record[InputColumn.NEXT_ACTION],
                action_size_plus_one,
            )

        if self.normalize:
            C2.set_net_and_init_net(net, init_net)
            state, _ = PreprocessorNet().normalize_dense_matrix(
                state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="state",
                split_expensive_feature_groups=True,
            )
            next_state, _ = PreprocessorNet().normalize_dense_matrix(
                next_state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="next_state",
                split_expensive_feature_groups=True,
            )
            if self.sorted_action_features is not None:
                action, _ = PreprocessorNet().normalize_dense_matrix(
                    action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="action",
                    split_expensive_feature_groups=True,
                )
                next_action, _ = PreprocessorNet().normalize_dense_matrix(
                    next_action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="next_action",
                    split_expensive_feature_groups=True,
                )
                if self.include_possible_actions:
                    possible_action_features, _ = PreprocessorNet().normalize_dense_matrix(
                        possible_action_features,
                        self.sorted_action_features,
                        self.action_normalization_parameters,
                        blobname_prefix="possible_action",
                        split_expensive_feature_groups=True,
                    )
                    possible_next_action_features, _ = PreprocessorNet().normalize_dense_matrix(
                        possible_next_action_features,
                        self.sorted_action_features,
                        self.action_normalization_parameters,
                        blobname_prefix="possible_next_action",
                        split_expensive_feature_groups=True,
                    )
            C2.set_net_and_init_net(None, None)

        if self.metrics_to_score:
            metrics_to_score_idxs = list(range(len(self.metrics_to_score)))
            missing_metric = self.create_const(init_net, "MISSING_METRIC", 0.0)
            metrics, metrics_presence = self.extract_float_features(
                net,
                InputColumn.METRICS,
                input_record[InputColumn.METRICS],
                metrics_to_score_idxs,
                missing_metric,
            )

        if self.use_time_since_first:
            time_since_first = net.Cast(
                net.ExpandDims(
                    input_record[InputColumn.TIME_SINCE_FIRST](), 1, dims=[1]
                ),
                net.NextScopedBlob("float_time_since_first"),
                to=core.DataType.FLOAT,
            )
            float_time_diff = net.Cast(
                net.ExpandDims(input_record[InputColumn.TIME_DIFF](), 1, dims=[1]),
                net.NextScopedBlob("float_time_diff"),
                to=core.DataType.FLOAT,
            )

            next_time_since_first = net.Add(
                [time_since_first, float_time_diff],
                net.NextScopedBlob("float_next_time_since_first"),
            )
            if self.time_since_first_normalization_parameters and self.normalize:
                C2.set_net_and_init_net(net, init_net)
                time_since_first, _ = PreprocessorNet().normalize_dense_matrix(
                    time_since_first,
                    [0],
                    {0: self.time_since_first_normalization_parameters},
                    blobname_prefix="time_since_first",
                    split_expensive_feature_groups=True,
                )
                next_time_since_first, _ = PreprocessorNet().normalize_dense_matrix(
                    next_time_since_first,
                    [0],
                    {0: self.time_since_first_normalization_parameters},
                    blobname_prefix="next_time_since_first",
                    split_expensive_feature_groups=True,
                )
                C2.set_net_and_init_net(None, None)

        output_schema = schema.Struct(
            (InputColumn.STATE_FEATURES, state),
            (InputColumn.STATE_FEATURES_PRESENCE, state_presence),
            (InputColumn.NEXT_STATE_FEATURES, next_state),
            (InputColumn.NEXT_STATE_FEATURES_PRESENCE, next_state_presence),
            (InputColumn.ACTION, action),
            (InputColumn.NEXT_ACTION, next_action),
        )

        if self.sorted_action_features:
            output_schema += schema.Struct(
                (InputColumn.ACTION_PRESENCE, action_presence),
                (InputColumn.NEXT_ACTION_PRESENCE, next_action_presence),
            )

        output_schema += schema.Struct(
            *(
                [
                    (col_name, input_record[col_name])
                    for col_name, _col_type in pass_through_columns
                ]
            )
        )

        if self.has_sequence_features:
            output_schema += schema.Struct(
                (
                    "state_sequence_features",
                    self.get_state_sequence_features_schema(
                        state_sequence_id_features, state_sequence_float_features
                    ),
                ),
                (
                    "next_state_sequence_features",
                    self.get_state_sequence_features_schema(
                        next_state_sequence_id_features,
                        next_state_sequence_float_features,
                    ),
                ),
            )

        if self.use_time_since_first:
            output_schema += schema.Struct(
                ("time_since_first", time_since_first),
                ("next_time_since_first", next_time_since_first),
            )

        if self.include_possible_actions:
            # Drop the "lengths" blob from possible_actions_mask since we know
            # it's just a list of [max_num_actions, max_num_actions, ...]
            output_schema += schema.Struct(
                (
                    InputColumn.POSSIBLE_ACTIONS_MASK,
                    input_record[InputColumn.POSSIBLE_ACTIONS_MASK]["values"],
                ),
                (
                    InputColumn.POSSIBLE_NEXT_ACTIONS_MASK,
                    input_record[InputColumn.POSSIBLE_NEXT_ACTIONS_MASK]["values"],
                ),
            )
            if self.sorted_action_features is not None:
                output_schema += schema.Struct(
                    (InputColumn.POSSIBLE_ACTIONS, possible_action_features),
                    (
                        InputColumn.POSSIBLE_ACTIONS_PRESENCE,
                        possible_action_features_presence,
                    ),
                    (InputColumn.POSSIBLE_NEXT_ACTIONS, possible_next_action_features),
                    (
                        InputColumn.POSSIBLE_NEXT_ACTIONS_PRESENCE,
                        possible_next_action_features_presence,
                    ),
                )

        if self.metrics_to_score:
            output_schema += schema.Struct((InputColumn.METRICS, metrics))

        net.set_output_record(output_schema)
        return FeatureExtractorNet(net, init_net)


class WorldModelFeatureExtractor(FeatureExtractorBase):
    """
    Extract:
    - State
    - Action
    - Next state
    - Reward
    - Not terminal
    """

    def __init__(
        self,
        seq_len,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Optional[
            Dict[int, NormalizationParameters]
        ] = None,
        discrete_action_names: Optional[List[str]] = None,
        normalize: Optional[bool] = True,
    ) -> None:
        self.seq_len = seq_len
        self.normalize = normalize
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters

        self.sorted_state_features, _ = sort_features_by_normalization(
            state_normalization_parameters
        )
        self.state_dim = get_num_output_features(self.state_normalization_parameters)
        self.state_feature_num = len(self.sorted_state_features)
        self.sorted_state_feature_start_indices = get_feature_start_indices(
            self.sorted_state_features, state_normalization_parameters
        )

        if action_normalization_parameters:
            self.sorted_action_features, _ = sort_features_by_normalization(
                action_normalization_parameters
            )
            self.action_dim = get_num_output_features(
                self.action_normalization_parameters
            )
            self.action_feature_num = len(self.sorted_action_features)
            self.sorted_action_feature_start_indices = get_feature_start_indices(
                self.sorted_action_features, action_normalization_parameters
            )
        else:
            self.sorted_action_features = None
            assert discrete_action_names is not None
            self.action_dim = len(discrete_action_names)
            self.action_feature_num = len(discrete_action_names)
            self.sorted_action_feature_start_indices = list(
                range(len(discrete_action_names))
            )

    def extract(self, ws, extract_record):
        fetch = partial(self.fetch, ws)
        state = rlt.FeatureVector(
            float_features=rlt.ValuePresence(
                value=fetch(extract_record.state_features),
                presence=fetch(extract_record.state_features_presence),
            )
        )
        if self.sorted_action_features:
            action = rlt.FeatureVector(
                float_features=rlt.ValuePresence(
                    value=fetch(extract_record.action),
                    presence=fetch(extract_record.action_presence),
                )
            )
        else:
            action = fetch(extract_record.action).byte()
        next_state = rlt.FeatureVector(
            float_features=rlt.ValuePresence(
                value=fetch(extract_record.next_state_features),
                presence=fetch(extract_record.next_state_features_presence),
            )
        )
        reward = fetch(extract_record.reward["values"])
        not_terminal = fetch(extract_record.not_terminal["values"]).float()
        # TODO: Replace with true time diff
        time_diff = torch.ones_like(reward).float()
        training_input = rlt.RawMemoryNetworkInput(
            state=state,
            reward=reward,
            time_diff=time_diff,
            action=action,
            not_terminal=not_terminal,
            next_state=next_state,
            step=None,
        )
        return rlt.RawTrainingBatch(training_input=training_input, extras=None)

    def create_net(self):
        net = core.Net("feature_extractor")
        init_net = core.Net("feature_extractor_init")
        missing_scalar = self.create_const(init_net, "MISSING_SCALAR", MISSING_VALUE)
        action_schema = (
            schema.List(map_schema())
            if self.sorted_action_features
            else schema.List(schema.Scalar())
        )

        pass_through_columns = [
            (InputColumn.NOT_TERMINAL, schema.List(schema.Scalar())),
            (InputColumn.REWARD, schema.List(schema.Scalar())),
        ]
        input_schema = schema.Struct(
            *(
                [
                    (InputColumn.STATE_FEATURES, schema.List(map_schema())),
                    (InputColumn.ACTION, action_schema),
                    (InputColumn.NEXT_STATE_FEATURES, schema.List(map_schema())),
                ]
                + pass_through_columns
            )
        )
        input_record = net.set_input_record(input_schema)

        state, state_presence = self.extract_float_features(
            net,
            "state",
            input_record[InputColumn.STATE_FEATURES].value,
            self.sorted_state_features,
            missing_scalar,
        )
        next_state, next_state_presence = self.extract_float_features(
            net,
            "next_state",
            input_record[InputColumn.NEXT_STATE_FEATURES].value,
            self.sorted_state_features,
            missing_scalar,
        )

        if self.sorted_action_features:
            action, action_presence = self.extract_float_features(
                net,
                InputColumn.ACTION,
                input_record[InputColumn.ACTION].value,
                self.sorted_action_features,
                missing_scalar,
            )
        else:
            action_size_plus_one = self.create_const(
                init_net,
                "action_size_plus_one",
                self.action_dim + 1,
                dtype=core.DataType.INT64,
            )
            action = self.read_actions_to_mask(
                net,
                InputColumn.ACTION,
                self.action_dim,
                input_record[InputColumn.ACTION].value,
                action_size_plus_one,
            )
            action_presence = action

        if self.normalize:
            C2.set_net_and_init_net(net, init_net)
            state, _ = PreprocessorNet().normalize_dense_matrix(
                state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="state",
                split_expensive_feature_groups=True,
            )
            next_state, _ = PreprocessorNet().normalize_dense_matrix(
                next_state,
                self.sorted_state_features,
                self.state_normalization_parameters,
                blobname_prefix="next_state",
                split_expensive_feature_groups=True,
            )
            if self.sorted_action_features is not None:
                action, _ = PreprocessorNet().normalize_dense_matrix(
                    action,
                    self.sorted_action_features,
                    self.action_normalization_parameters,
                    blobname_prefix="action",
                    split_expensive_feature_groups=True,
                )
            C2.set_net_and_init_net(None, None)

        output_schema = schema.Struct(
            *(
                [
                    (InputColumn.STATE_FEATURES, state),
                    (InputColumn.STATE_FEATURES_PRESENCE, state_presence),
                    (InputColumn.NEXT_STATE_FEATURES, next_state),
                    (InputColumn.NEXT_STATE_FEATURES_PRESENCE, next_state_presence),
                    (InputColumn.ACTION, action),
                    (InputColumn.ACTION_PRESENCE, action_presence),
                ]
                + [
                    (col_name, input_record[col_name])
                    for col_name, _col_type in pass_through_columns
                ]
            )
        )
        net.set_output_record(output_schema)
        return FeatureExtractorNet(net, init_net)
