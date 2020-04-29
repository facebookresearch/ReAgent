#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

CONFIG=reagent/workflow/sample_configs/sac_pendulum_offline.yaml

# gather data and store as pickle
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym $CONFIG

# run through timeline operator
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator $CONFIG

# train and evaluate
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.train_and_evaluate_gym $CONFIG

echo "End-to-end test passed!"
