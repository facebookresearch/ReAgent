#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

pip install -e .

mkdir -p cartpole_discrete
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_small_v0.json -f cartpole_discrete/training_data.json --seed 0

spark-submit \
  --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
  "$(cat ml/rl/workflow/sample_configs/discrete_action/timeline.json)"

mkdir -p training_data
cat cartpole_discrete_training/part* > training_data/cartpole_discrete_timeline.json
cat cartpole_discrete_eval/part* > training_data/cartpole_discrete_timeline_eval.json

# Remove the output data folder
rm -Rf cartpole_discrete_training cartpole_discrete_eval

python ml/rl/workflow/create_normalization_metadata.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

mkdir -p outputs
rm -Rf outputs/model_* outputs/*.txt
python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

# Evaluate
python ml/rl/test/workflow/eval_cartpole.py -m outputs/model_* --softmax_temperature=0.35 --log_file=outputs/eval_output.txt

# Reach at least 120 in cart-pole
eval_res=$(awk '/Achieved an average reward score of /{print $7}' outputs/eval_output.txt)
pass=$(echo "$eval_res > 120.0" | bc)
if [ "$pass" -eq 0 ]; then exit 1;  fi
