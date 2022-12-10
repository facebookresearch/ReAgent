from dataclasses import replace

import torch
from reagent.core.types import CBInput


def zero_out_skipped_obs_weights(
    batch: CBInput, model_actions: torch.Tensor
) -> CBInput:
    """
    Return a copy of the input batch, but with weights zero-ed out where the logged action and the model action
        don't match.
    """
    current_weight = batch.weight
    if current_weight is None:
        current_weight = torch.ones(len(batch), 1, device=batch.device)
    logged_actions = batch.action
    assert logged_actions is not None
    assert current_weight.shape == logged_actions.shape, (
        current_weight.shape,
        logged_actions.shape,
    )
    assert logged_actions.shape == model_actions.shape, (
        logged_actions.shape,
        model_actions.shape,
    )
    new_batch = replace(
        batch, weight=current_weight * (logged_actions == model_actions)
    )
    return new_batch
