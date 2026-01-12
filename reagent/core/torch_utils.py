#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def dict_to_tensor(batch: Dict[str, np.ndarray], device: str = "cpu"):
    return {k: torch.tensor(v).to(device) for k, v in batch.items()}


def rescale_torch_tensor(
    tensor: torch.Tensor,
    new_min: torch.Tensor,
    new_max: torch.Tensor,
    prev_min: torch.Tensor,
    prev_max: torch.Tensor,
):
    """
    Rescale column values in N X M torch tensor to be in new range.
    Each column m in input tensor will be rescaled from range
    [prev_min[m], prev_max[m]] to [new_min[m], new_max[m]]
    """
    assert tensor.shape[1] == new_min.shape[1] == new_max.shape[1]
    assert tensor.shape[1] == prev_min.shape[1] == prev_max.shape[1]
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((tensor - prev_min) / prev_range) * new_range + new_min


def stack(mems):
    """
    Stack a list of tensors
    Could use torch.stack here but torch.stack is much slower
    than torch.cat + view
    Submitted an issue for investigation:
    https://github.com/pytorch/pytorch/issues/22462

    FIXME: Remove this function after the issue above is resolved
    """
    shape = (-1, *mems[0].shape)
    return torch.cat(mems).view(*shape)


def export_module_to_buffer(module) -> BytesIO:
    # traced_script_module = torch.jit.trace(module, module.input_prototype())
    write_buffer = BytesIO()
    torch.jit.save(module, write_buffer)
    return write_buffer


def softmax(x, temperature):
    """Compute softmax values for each sets of scores in x."""
    x = x / temperature
    return torch.nn.functional.softmax(x, dim=1)


def masked_softmax(x, mask: float, temperature):
    """Compute softmax values for each sets of scores in x."""
    x = x / temperature
    mask_min_x = x - ((1.0 - mask) * 1e20)
    mask_min_x -= torch.max(mask_min_x, dim=1, keepdim=True)[0]
    e_x = torch.exp(mask_min_x)
    e_x *= mask
    out = e_x / e_x.sum(dim=1, keepdim=True)

    # Set NaN values to 0 (NaN happens when a full mask row is passed in)
    out[out != out] = 0
    return out


def gather(data, index_2d):
    """
    Gather data alongs the second dim. Assume data is 3d with shape (batch_size, dim1, dim2),
    and index_2d's shape is (batch_size, dim1).
    output[i][j] = data[i][index_2d[i][j]]

    This function does not require data, output, or index_2d having the same shape, which
     is mandated by torch.gather.
    """
    batch_size = data.shape[0]
    data_dim = data.shape[2]
    index_len = index_2d.shape[1]
    device = data.device
    res = data[
        torch.arange(batch_size, device=device).repeat_interleave(
            # index_len has to be moved to the device explicitly, otherwise
            # error will throw during jit.trace
            torch.tensor([index_len], device=device)
        ),
        index_2d.flatten(),
    ].view(batch_size, index_len, data_dim)
    return res


def get_device(model):
    return next(model.parameters()).device


def split_sequence_keyed_jagged_tensor(
    x: KeyedJaggedTensor, num_steps: int
) -> List[KeyedJaggedTensor]:
    """
    Input:
    x (KeyedJaggedTensor): represents a batch of sequential sparse data.
        Analogous to a batch of sequential dense data with shape:
        batch_size x num_steps x num_dense_feature

    Return:
        Split data into individual steps and return a list of KeyedJaggedTensor
        (the length of the list equals to num_steps)

    Example:
    Input KeyedJaggedTensor (x):
        x = KeyedJaggedTensor(
            keys=["Key0", "Key1", "Key2"],
            values=[V0, V1, V2, V3, V4, V5, V6, V7, V8, V9]
            lengths=[2, 0, 1, 1, 1, 1, 3, 0, 0, 1, 0, 0]
        )
    which represents a minibatch of 2 data points with three keys and two steps:
                   data0_step0    data0_step1    data1_step0     data1_step1
        "Key0"       [V0,V1]          None           [V2]            [V3]
        "Key1"         [V4]           [V5]        [V6,V7,V8]         None
        "Key2"         None           [V9]           None            None

    It will be split and returned as a list of two KeyedJaggedTensor:
    [
        # step 0
        KeyedJaggedTensor(
            keys=["Key0", "Key1", "Key2"],
            values=[V0, V1, V2, V4, V6, V7, V8]
            lengths=[2, 1, 1, 3, 0, 0]
        ),
        # step 1
        KeyedJaggedTensor(
            keys=["Key0", "Key1", "Key2"],
            values=[V3, V5, V9]
            lengths=[0, 1, 1, 0, 1, 0]
        )
    ]
    """
    keys = x.keys()
    has_weights = x._weights is not None
    split_dict = {}
    for i in range(num_steps):
        split_dict[i] = {}
    for key in keys:
        keyed_x: JaggedTensor = x[key]
        weights = keyed_x._weights
        values = keyed_x.values()
        lengths = keyed_x.lengths()

        # Because len(lengths) == batch_size * num_steps
        assert len(lengths) % num_steps == 0

        splitted_values = torch.split(values, lengths.tolist())
        if has_weights:
            # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
            splitted_weights = torch.split(weights, lengths.tolist())
        for i in range(num_steps):
            split_dict[i][key] = (
                lengths[i::num_steps],
                torch.cat(splitted_values[i::num_steps]),
                # pyre-fixme[61]: `splitted_weights` is undefined, or not always
                #  defined.
                torch.cat(splitted_weights[i::num_steps]) if has_weights else None,
            )

    result: List[KeyedJaggedTensor] = []
    for i in range(num_steps):
        result.append(
            KeyedJaggedTensor(
                keys=keys,
                lengths=torch.cat([split_dict[i][k][0] for k in keys]),
                values=torch.cat([split_dict[i][k][1] for k in keys]),
                weights=(
                    torch.cat([split_dict[i][k][2] for k in keys])
                    if has_weights
                    else None
                ),
            )
        )
    return result


def reorder_data_kjt(x: KeyedJaggedTensor, indices: torch.Tensor):
    """
    Reorder the data for each key in a KeyedJaggedTensor

    Input:
        indices: Long tensor represents the order of returned data for each key

    Example:
    Input KeyedJaggedTensor (x):
        x = KeyedJaggedTensor(
            keys=["Key0", "Key1"],
            values=[V0, V1, V2, V3, V4, V5, V6]
            lengths=[2, 0, 1, 1, 1, 2]
        )
    which represents data:
                   data0       data1      data2
        "Key0"    [V0,V1]       None       [V2]
        "Key1"      [V3]        [V4]      [V5,V6]

    If we wish to order data as [data2, data1, data0], then this function will return
                   data0       data1      data2
        "Key0"     [V2]         None     [V0, V1]
        "Key1"    [V5,V6]       [V4]       [V3]
    """
    num_keys = len(x.keys())
    num_data = len(indices)
    assert len(x.lengths()) == num_keys * num_data, (
        "The num of data indicated by input arg indices does not match with input KeyedJaggedTensor"
    )

    acc_lengths_per_key = torch.cumsum(torch.tensor(x.length_per_key()), dim=0)
    values_per_key = torch.tensor_split(x.values(), acc_lengths_per_key)[:-1]
    val_lens_per_key = torch.chunk(x.lengths(), num_keys)
    splitted_vals_per_key = [
        torch.tensor_split(x, torch.cumsum(y, dim=0))[:-1]
        for x, y in zip(values_per_key, val_lens_per_key)
    ]

    # Reorder values, lengths, and weights *WITHIN each key*
    reordered_vals = torch.cat(
        [torch.cat([x[y] for y in indices.tolist()]) for x in splitted_vals_per_key]
    )
    reordered_lengths = torch.cat([x[indices] for x in val_lens_per_key])
    if x.weights_or_none() is not None:
        weights_per_key = torch.tensor_split(x.weights(), acc_lengths_per_key)[:-1]
        splitted_weights_per_key = [
            torch.tensor_split(x, torch.cumsum(y, dim=0))[:-1]
            for x, y in zip(weights_per_key, val_lens_per_key)
        ]
        reordered_weights = torch.cat(
            [
                torch.cat([x[y] for y in indices.tolist()])
                for x in splitted_weights_per_key
            ]
        )
    else:
        reordered_weights = None

    res = KeyedJaggedTensor(
        keys=x.keys(),
        lengths=reordered_lengths,
        values=reordered_vals,
        weights=reordered_weights,
    )
    return res


def shift_kjt_by_one(x: KeyedJaggedTensor):
    """
    Shift the data by one for each key in a KeyedJaggedTensor
    The last data will then always have no value

    Example:
    Input KeyedJaggedTensor (x):
        x = KeyedJaggedTensor(
            keys=["Key0", "Key1"],
            values=[V0, V1, V2, V3, V4, V5, V6]
            lengths=[2, 0, 1, 1, 1, 2]
        )
    which represents data:
                   data0       data1      data2
        "Key0"    [V0,V1]       None       [V2]
        "Key1"      [V3]        [V4]      [V5,V6]

    If we wish to shift data by one, then this function will return
                   data0       data1      data2
        "Key0"     None         [V2]       None
        "Key1"     [V4]        [V5,V6]     None
    """
    num_keys = len(x.keys())
    acc_lengths_per_key = torch.cumsum(torch.tensor(x.length_per_key()), dim=0)
    values_per_key = torch.tensor_split(x.values(), acc_lengths_per_key)[:-1]
    val_lens_per_key = torch.chunk(x.lengths(), num_keys)

    # Shift values, lengths, and weights *WITHIN each key*
    shifted_vals = torch.cat(
        [x[y[0] :] for x, y in zip(values_per_key, val_lens_per_key)]
    )
    shifted_lengths = torch.cat(
        [
            torch.cat([x[1:], torch.tensor([0], device=x.device)])
            for x in val_lens_per_key
        ]
    )
    if x.weights_or_none() is not None:
        weights_per_key = torch.tensor_split(x.weights(), acc_lengths_per_key)[:-1]
        shifted_weights = torch.cat(
            [x[y[0] :] for x, y in zip(weights_per_key, val_lens_per_key)]
        )
    else:
        shifted_weights = None

    res = KeyedJaggedTensor(
        keys=x.keys(),
        lengths=shifted_lengths,
        values=shifted_vals,
        weights=shifted_weights,
    )
    return res
