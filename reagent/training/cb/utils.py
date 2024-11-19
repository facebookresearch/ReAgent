#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
from dataclasses import replace
from typing import List, Optional, Union

import torch
from reagent.core.types import CBInput


logger = logging.getLogger(__name__)


def add_chosen_arm_features(
    batch: Union[CBInput, List[CBInput]],
) -> Union[CBInput, List[CBInput]]:
    """
    Add the features for chosen arms to the batch.
    For joint models:
        - Both input and output are`CBInput` objects
        - batch.all_arm_features is a 3D tensor of shape (batch_size, num_arms, arm_dim)
        - `batch.action` is used to choose which features to extract.
    For joint models:
         - Both input and output are `List[CBInput]` objects of len `num_arms`
         - batch.all_arm_features is a 2D tensor of shape (batch_size, arm_dim)
         - This function just extracts the `all_arm_features` attribute from batch and packages them back into a list

    Args:
        batch: A batch of input data.
            Attributes of the batch:
                all_arm_features: Tensor with features of all available arms.
                action: 2D Tensor of shape (batch_size, 1) with dtype long. For each observation
                    it holds the index of the chosen arm.
    Returns:
        For joint models:
            A 2D Tensor of shape (batch_size, arm_dim) with features of chosen arms.
        For disjoint models:
            A list of 2D Tensors of shape (batch_size, arm_dim)
    """
    if isinstance(batch, CBInput):
        assert batch.context_arm_features.ndim == 3
        assert batch.action is not None
        batch = replace(
            batch,
            features_of_chosen_arm=torch.gather(
                batch.context_arm_features,
                1,
                batch.action.unsqueeze(-1).expand(
                    -1, 1, batch.context_arm_features.shape[2]
                ),
            ).squeeze(1),
        )
        if batch.arms is not None:
            assert batch.action is not None
            batch = replace(
                batch,
                chosen_arm_id=torch.gather(batch.arms, 1, batch.action),
            )
        return batch
    elif isinstance(batch, list):
        assert isinstance(batch[0], CBInput)
        assert batch[0].context_arm_features.ndim == 2
        return [
            replace(b, features_of_chosen_arm=b.context_arm_features) for b in batch
        ]
    else:
        raise ValueError(
            f"Unexpected input type {type(batch)} for _add_chosen_arm_features"
        )


def argmax_random_tie_breaks(
    scores: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Given a 2D tensor of scores, return the indices of the max score for each row.
    If there are ties inside a row, uniformly randomize among the ties.
    IMPORTANT IMPLEMENTATION DETAILS:
        1. Randomization is implemented consistently across all rows. E.g. if several columns
            are tied on 2 different rows, we will return the same index for each of these rows.

    Args:
        scores: A 2D tensor of scores
        mask [Optional]: A 2D score presence mask. If missing, assuming that all scores are unmasked.
    """
    # This function only works for 2D tensor
    assert scores.ndim == 2

    # Permute the columns
    num_cols = scores.size(1)
    random_col_indices = torch.randperm(num_cols)
    permuted_scores = torch.index_select(scores, 1, random_col_indices)
    if mask is not None:
        permuted_mask = torch.index_select(mask, 1, random_col_indices)
        permuted_scores = torch.masked.as_masked_tensor(
            permuted_scores, permuted_mask.bool()
        )

    # Find the indices of the maximum elements in the random permutation
    max_indices_in_permuted_data = torch.argmax(permuted_scores, dim=1)

    if mask is not None:
        # pyre-fixme[16]: `Tensor` has no attribute `get_data`.
        max_indices_in_permuted_data = max_indices_in_permuted_data.get_data().long()

    # Use the random permutation to get the original indices of the maximum elements
    argmax_indices = random_col_indices[max_indices_in_permuted_data]

    return argmax_indices


def get_model_actions(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    randomize_ties: bool = False,
) -> torch.Tensor:
    """
    Given a tensor of scores, get the indices of chosen actions.
    Chosen actions are the score argmax (within each row), subject to optional mask.
    if `randomize_ties`=True, we will also randomize the order of tied actions with
        maximum values. This has computational cost compared to not randomizing (use 1st index)
    """
    if mask is None:
        if randomize_ties:
            model_actions = argmax_random_tie_breaks(scores).reshape(-1, 1)
        else:
            model_actions = torch.argmax(scores, dim=1).reshape(-1, 1)
    else:
        if randomize_ties:
            model_actions = argmax_random_tie_breaks(scores, mask)
        else:
            # mask out non-present arms
            scores_masked = torch.masked.as_masked_tensor(scores, mask.bool())
            model_actions = (
                # pyre-fixme[16]: `Tensor` has no attribute `get_data`.
                torch.argmax(scores_masked, dim=1).get_data().reshape(-1, 1)
            )
    return model_actions
