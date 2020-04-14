#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Build timeline package (only need to do this first time)
mvn -f preprocessing/pom.xml clean package

# Clear last run's spark data (in case of interruption) and run timelime on pre-timeline data
function finish {
  rm -Rf spark-warehouse derby.log metastore_db preprocessing/spark-warehouse preprocessing/metastore_db preprocessing/derby.log
}
trap finish EXIT

# Run timelime on pre-timeline data
/usr/local/spark/bin/spark-submit \
  --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
  "`cat reagent/workflow/sample_configs/discrete_action/timeline.json`"

mkdir training_data
mv cartpole_discrete_timeline/part* training_data/cartpole_training_data.json

# Remove the output data folder
rm -Rf cartpole_discrete_timeline
