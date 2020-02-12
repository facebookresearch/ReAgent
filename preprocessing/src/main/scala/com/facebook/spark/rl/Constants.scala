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

  val RANKING_DATA_COLUMN_NAMES = Array(
    "ds",
    "mdp_id",
    "sequence_number",
    "slate_reward",
    "item_reward",
    "action",
    "action_probability",
    "state_features",
    "state_sequence_features",
    "next_action",
    "next_state_features",
    "next_state_sequence_features"
  );

  val SPARSE_DATA_COLUMN_NAMES = Array(
    "state_id_list_features",
    "state_id_score_list_features",
    "next_state_id_list_features",
    "next_state_id_score_list_features"
  );

  val SPARSE_ACTION_COLUMN_NAMES = Array(
    "action_id_list_features",
    "action_id_score_list_features",
    "next_action_id_list_features",
    "next_action_id_score_list_features"
  );

  val DEFAULT_REWARD_COLUMNS = List[String](
    "reward",
    "metrics"
  );

  val DEFAULT_REWARD_TYPES = Map(
    "reward" -> "double",
    "metrics" -> "map<string,double>"
  );

  val DEFAULT_EXTRA_FEATURE_COLUMNS = List[String]()

}
