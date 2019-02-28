#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds Horizon and runs basic tests.

pip uninstall -y horizon

thrift --gen py --out . ml/rl/thrift/core.thrift

# Installing from current directory, any update will be reflected system-wide
pip install -e .
pytest

# Build the spark package
mvn -f preprocessing/pom.xml clean package

mkdir cartpole_discrete
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0.json -f cartpole_discrete/training_data.json

/usr/local/spark/bin/spark-submit \
  --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
  "`cat ml/rl/workflow/sample_configs/discrete_action/timeline.json`"

mkdir training_data
cat cartpole_discrete_training/part* > training_data/cartpole_discrete_timeline.json
cat cartpole_discrete_eval/part* > training_data/cartpole_discrete_timeline_eval.json

# Remove the output data folder
rm -Rf cartpole_discrete_training cartpole_discrete_eval

python ml/rl/workflow/create_normalization_metadata.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

mkdir outputs
python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

# Evaluate
python ml/rl/test/workflow/eval_cartpole.py -m outputs/predictor*