#!/usr/bin/env bash

# Build timeline package
mvn -f preprocessing/pom.xml package

# Clear last run's spark data (in case of interruption) and run timelime on pre-timeline data
function finish {
  rm -Rf spark-warehouse training_data derby.log metastore_db
}
trap finish EXIT

# Remove the output data
rm -Rf cartpole_discrete_timeline

# Run timelime on pre-timeline data
/usr/local/spark/bin/spark-submit \
  --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
  "`cat ml/rl/workflow/sample_configs/discrete_action/timeline.json`"
