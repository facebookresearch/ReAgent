#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from .discrete_c51dqn import DiscreteC51DQN
from .discrete_crr import DiscreteCRR
from .discrete_dqn import DiscreteDQN
from .discrete_qrdqn import DiscreteQRDQN

__all__ = ["DiscreteC51DQN", "DiscreteDQN", "DiscreteQRDQN", "DiscreteCRR"]
