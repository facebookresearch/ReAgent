#!/usr/bin/env bash

# Build timeline package (only need to do this first time)
mvn -f preprocessing/pom.xml package

# Move the training data to a directory (expected by Spark)
mkdir ~/cartpole_discrete
cp ~/ cartpole_discrete/training_data.json ~/cartpole_discrete

# Run timelime on pre-timeline data
/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar "`cat home/cartpole_training_data.json`"
