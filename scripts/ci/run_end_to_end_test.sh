#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

chmod +x ./reagent/workflow/cli.py

# gather data and upload to hive
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym reagent/workflow/sample_configs/cartpole_discrete_dqn_offline.yaml
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.upload_to_hive reagent/workflow/sample_configs/cartpole_discrete_dqn_offline.yaml

# run through timeline operator
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator reagent/workflow/sample_configs/cartpole_discrete_dqn_offline.yaml

# train and evaluate
./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.train_and_evaluate_gym reagent/workflow/sample_configs/cartpole_discrete_dqn_offline.yaml

echo "End-to-end test passed!"
