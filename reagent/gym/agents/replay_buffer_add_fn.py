#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import reagent.types as rlt
import torch
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


def replay_buffer_add_fn(
    replay_buffer: ReplayBuffer,
    obs: Any,
    actor_output: rlt.ActorOutput,
    reward: float,
    terminal: bool,
    possible_actions_mask: Optional[torch.Tensor] = None,
) -> None:
    """ Simply adds transition into buffer after converting to numpy """
    action = actor_output.action.numpy()
    log_prob = actor_output.log_prob.numpy()
    if possible_actions_mask is None:
        possible_actions_mask = torch.ones_like(actor_output.action).to(torch.bool)
    possible_actions_mask = possible_actions_mask.numpy()
    replay_buffer.add(obs, action, reward, terminal, possible_actions_mask, log_prob)
