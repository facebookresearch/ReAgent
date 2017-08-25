'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package rlmodels
# Module rl2_caffe2.rlmodel

from .rlmodel_helper import add_parameter_update_ops
from .rlmodel_helper import sample_memories
from .rlmodel_helper import GRAD_OPTIMIZER, OPTIMIZER_DICT, MODEL_T, MODEL_T_DICT

from .rlmodel_base import RLNN
from .dqn import DQN_rlnn
from .actor_critic import ActorCritic_rlnn
