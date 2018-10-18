#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

mkdir cartpole_discrete

python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0_100_eps.json -f cartpole_discrete/training_data.json
