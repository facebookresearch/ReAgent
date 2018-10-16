#!/usr/bin/env bash

# Build timeline package (only need to do this first time)
mvn -f preprocessing/pom.xml package

# Clear last run's spark data (in case of interruption) and run timelime on pre-timeline data
function finish {
  rm -Rf spark-warehouse training_data derby.log metastore_db
}
trap finish EXIT

# Remove the output data
rm -Rf cartpole_discrete_timeline

/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar '
{
  "timeline": {
    "startDs": "2019-01-01",
    "endDs": "2019-01-01",
    "addTerminalStateRow": false,
    "actionDiscrete": true,
    "inputTableName": "cartpole_discrete",
    "outputTableName": "cartpole_discrete_timeline"
  },
  "query": {
    "discountFactor": 0.99,
    "tableSample": 1,
    "maxQLearning": true,
    "useNonOrdinalRewardTimeline": false,
    "actions": [
      "4",
      "5"
    ]
  }
}
'
