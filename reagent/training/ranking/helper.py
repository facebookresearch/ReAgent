#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Optional

import torch
from reagent.parameters_seq2slate import IPSClamp, IPSClampMethod


def ips_clamp(impt_smpl, ips_clamp: Optional[IPSClamp]):
    if not ips_clamp:
        return impt_smpl.clone()
    if ips_clamp.clamp_method == IPSClampMethod.UNIVERSAL:
        return torch.clamp(impt_smpl, 0, ips_clamp.clamp_max)
    elif ips_clamp.clamp_method == IPSClampMethod.AGGRESSIVE:
        return torch.where(
            impt_smpl > ips_clamp.clamp_max, torch.zeros_like(impt_smpl), impt_smpl
        )
