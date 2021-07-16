#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch

logger = logging.getLogger(__name__)


def get_valid_actions_from_imitator(imitator, input, drop_threshold):
    """Create mask for non-viable actions under the imitator."""
    if isinstance(imitator, torch.nn.Module):
        # pytorch model
        imitator_outputs = imitator(input.float_features)
        on_policy_action_probs = torch.nn.functional.softmax(imitator_outputs, dim=1)
    else:
        # sci-kit learn model
        on_policy_action_probs = torch.tensor(imitator(input.float_features.cpu()))

    filter_values = (
        on_policy_action_probs / on_policy_action_probs.max(keepdim=True, dim=1)[0]
    )
    return (filter_values >= drop_threshold).float()
