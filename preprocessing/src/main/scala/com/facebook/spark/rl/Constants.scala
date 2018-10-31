// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

object Constants {
  val TRAINING_DATA_COLUMN_NAMES = Array(
    "ds",
    "mdp_id",
    "state_features",
    "action",
    "action_probability",
    "reward",
    "next_state_features",
    "next_action",
    "sequence_number",
    "sequence_number_ordinal",
    "time_diff",
    "possible_actions",
    "possible_next_actions",
    "metrics"
  );

}
