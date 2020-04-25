#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .policy import Policy
from .random_policies import DiscreteRandomPolicy


__all__ = ["Policy", "DiscreteRandomPolicy"]
