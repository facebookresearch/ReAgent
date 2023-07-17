from dataclasses import replace
from typing import Optional

import torch
from reagent.core.types import CBInput


def add_importance_weights(
    batch: CBInput,
    model_actions: torch.Tensor,
    max_importance_weight: Optional[float] = None,
) -> CBInput:
    """
    Add importnace weights to the batch. Importance weights are:
    1. Zero ut where the logged action and the model action don't match.
    2. (1/logged_action_probability) where the logged action and the model action match.
    3. Optionally clipped at max value `max_importance_weight`
    """
    logged_actions = batch.action
    assert logged_actions is not None
    assert logged_actions.shape == model_actions.shape, (
        logged_actions.shape,
        model_actions.shape,
    )
    if batch.action_log_probability is not None:
        logged_action_prob = torch.exp(batch.action_log_probability)
    else:
        # if probabilities weren't logged, assume uniform random selection
        if batch.arm_presence is not None:
            slate_sizes = batch.arm_presence.sum(1, keepdim=True)
        else:
            # if arm presence wasn't specified, assume all arms are present
            slate_sizes = (
                torch.ones_like(logged_actions) * batch.context_arm_features.shape[1]
            )  # context_arm_features has shape (batch_size, num_arms, feature_dim)
        logged_action_prob = torch.ones_like(slate_sizes) / slate_sizes

    # importance weight inversely proportional to probabilities of logged actions
    importance_weights = torch.ones_like(logged_action_prob) / logged_action_prob
    if max_importance_weight is not None:
        importance_weights = torch.clamp(importance_weights, max=max_importance_weight)
    new_batch = replace(
        batch,
        importance_weight=(logged_actions == model_actions) * importance_weights,
    )
    return new_batch
