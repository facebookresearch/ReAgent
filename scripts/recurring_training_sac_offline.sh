#!/bin/bash

set -x -e

rm -f /tmp/file_system_publisher
rm -Rf test_warmstart model_* pl_log* runs

CONFIG=reagent/workflow/sample_configs/sac_pendulum_offline.yaml

python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym_random "$CONFIG"
rm -Rf spark-warehouse derby.log metastore_db
python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator "$CONFIG"
python ./reagent/workflow/cli.py run reagent.workflow.training.identify_and_train_network "$CONFIG"
python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.evaluate_gym "$CONFIG"

for _ in {0..30}
do
python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym_predictor "$CONFIG"
rm -Rf spark-warehouse derby.log metastore_db
python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator "$CONFIG"
python ./reagent/workflow/cli.py run reagent.workflow.training.identify_and_train_network "$CONFIG"
python ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.evaluate_gym "$CONFIG"
done
