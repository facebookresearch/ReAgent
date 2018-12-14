// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

object Query {

  def getDiscreteQuery(config: QueryConfiguration): String = {
    var query = """
    SELECT
        mdp_id,
        sequence_number,
        action_probability as propensity,
        state_features,
        CASE action
    """
    for (i <- 0 until config.actions.length) {
      query = query.concat(
        s"WHEN '${config.actions(i)}' THEN CAST(${i} AS BIGINT) "
      )
    }
    query = query.concat("""
      END AS action,
      reward,
      next_state_features,
      time_diff,
    """)

    query = query.concat("ARRAY(")
    for (i <- 0 until config.actions.length) {
      val putComma = if (i > 0) "," else ""
      query = query
        .concat(
          s"""${putComma}CAST(
          ARRAY_CONTAINS(possible_actions, '${config.actions(i)}') AS BIGINT)"""
        )
    }
    query = query.concat(") AS possible_actions,\n")

    query = query.concat("ARRAY(")
    for (i <- 0 until config.actions.length) {
      val putComma = if (i > 0) "," else ""
      query = query
        .concat(
          s"""${putComma}CAST(
          ARRAY_CONTAINS(possible_next_actions, '${config.actions(i)}') AS BIGINT)"""
        )
    }
    query = query.concat(") AS possible_next_actions,\n")

    query = query.concat("CASE next_action ")
    for (i <- 0 until config.actions.length) {
      query = query.concat(
        s"WHEN '${config.actions(i)}' THEN CAST(${i} AS BIGINT) "
      )
    }
    val s_num_actions = config.actions.length.toString
    query = query.concat(s"ELSE CAST(${s_num_actions} AS BIGINT) END AS next_action")

    query = query.concat(s"""
      , metrics
    """).stripMargin
    return query
  }

  def getContinuousQuery(config: QueryConfiguration): String =
    return s"""
    SELECT
      mdp_id,
      sequence_number,
      state_features,
      action,
      action_probability as propensity,
      reward,
      next_state_features,
      next_action,
      time_diff,
      possible_actions,
      possible_next_actions,
      metrics
    """.stripMargin
}
