#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

if [[ -z "${CONFIG}" ]]
then
    echo "Config path is not defined!"
    exit 1
else
    echo "Using config path: "
    echo "$CONFIG"
fi


# gather data and store as pickle
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym "$CONFIG"

# run through timeline operator
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator "$CONFIG"

# train and evaluate
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.train_and_evaluate_gym "$CONFIG"

echo "End-to-end test passed for config: "
echo "$CONFIG"
