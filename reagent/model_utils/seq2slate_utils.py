#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


PADDING_SYMBOL = 0
DECODER_START_SYMBOL = 1


class Seq2SlateMode(Enum):
    RANK_MODE = "rank"
    PER_SEQ_LOG_PROB_MODE = "per_sequence_log_prob"
    PER_SYMBOL_LOG_PROB_DIST_MODE = "per_symbol_log_prob_dist"
    DECODE_ONE_STEP_MODE = "decode_one_step"
    ENCODER_SCORE_MODE = "encoder_score_mode"


def subsequent_mask(size, device):
    """
    Mask out subsequent positions. Mainly used in the decoding process,
    in which an item should not attend subsequent items.
    """
    attn_shape = (1, size, size)
    subsequent_mask = (
        1 - torch.triu(torch.ones(*attn_shape, device=device), diagonal=1)
    ).type(torch.int8)
    return subsequent_mask


def subsequent_and_padding_mask(tgt_in_idx):
    """ Create a mask to hide padding and future items """
    # tgt_in_idx shape: batch_size, seq_len

    # tgt_tgt_mask shape: batch_size, 1, seq_len
    tgt_tgt_mask = (tgt_in_idx != PADDING_SYMBOL).unsqueeze(-2).type(torch.int8)
    # subseq_mask shape: 1, seq_len, seq_len
    subseq_mask = subsequent_mask(tgt_in_idx.size(-1), tgt_in_idx.device)
    # tgt_tgt_mask shape: batch_size, seq_len, seq_len
    tgt_tgt_mask = tgt_tgt_mask & subseq_mask
    return tgt_tgt_mask


def clones(module, N):
    """
    Produce N identical layers.

    :param module: nn.Module class
    :param N: number of copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask, d_k):
    """ Scaled Dot Product Attention """
    # mask shape: batch_size x 1 x seq_len x seq_len

    # scores shape: batch_size x num_heads x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    # p_attn shape: batch_size x num_heads x seq_len x seq_len
    p_attn = F.softmax(scores, dim=3)
    # attn shape: batch_size x num_heads x seq_len x d_k
    attn = torch.matmul(p_attn, value)
    return attn, p_attn


def per_symbol_to_per_seq_log_probs(per_symbol_log_probs, tgt_out_idx):
    """ Gather per-symbol log probabilities into per-seq log probabilities """
    # per_symbol_log_probs shape: batch_size, seq_len, candidate_size
    # tgt_out_idx shape: batch_size, seq_len
    # per_symbol_log_probs is log probability of each symbol in the tgt_out_idx
    # shape: batch_size, seq_len
    log_probs = torch.gather(per_symbol_log_probs, 2, tgt_out_idx.unsqueeze(2)).squeeze(
        2
    )
    # shape: batch_size, 1
    return log_probs.sum(dim=1, keepdim=True)


def per_symbol_to_per_seq_probs(per_symbol_probs, tgt_out_idx):
    """ Gather per-symbol probabilities into per-seq probabilities """
    # per_symbol_probs shape: batch_size, seq_len, candidate_size
    # tgt_out_idx shape: batch_size, seq_len
    # output shape: batch_size, 1
    return torch.prod(
        torch.gather(per_symbol_probs, 2, tgt_out_idx.unsqueeze(-1)).squeeze(),
        dim=-1,
        keepdim=True,
    )
