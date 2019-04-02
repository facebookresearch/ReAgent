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
    "time_since_first",
    "possible_actions",
    "possible_next_actions",
    "metrics"
  );

  val SPARSE_DATA_COLUMN_NAMES = Array(
    "state_id_list_features",
    "state_id_score_list_features",
    "next_state_id_list_features",
    "next_state_id_score_list_features"
  );

}
