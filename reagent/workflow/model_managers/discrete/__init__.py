#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .discrete_c51dqn import DiscreteC51DQN
from .discrete_dqn import DiscreteDQN
from .discrete_qrdqn import DiscreteQRDQN


__all__ = ["DiscreteC51DQN", "DiscreteDQN", "DiscreteQRDQN"]
