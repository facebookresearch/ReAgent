// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class MultiStepTimelineConfiguration(startDs: String,
                                          endDs: String,
                                          addTerminalStateRow: Boolean,
                                          actionDiscrete: Boolean,
                                          inputTableName: String,
                                          outputTableName: String,
                                          evalTableName: String,
                                          numOutputShards: Int,
                                          steps: Int)

/**
  * Given table of state, action, mdp_id, sequence_number, reward, possible_next_actions
  * return the table needed for reinforcement learning (MDP: Markov Decision Process)
  * mdp_id, state_features, action, reward, next_state_features, next_action,
  * sequence_number, sequence_number_ordinal, time_diff, possible_next_actions.
  * Shuffles the results.
  * Reference:
  * https://our.intern.facebook.com/intern/wiki/Reinforcement-learning/
  *
  * Args:
  * input_table: string, input table name
  *
  * output_table: string, output table name
  *
  * action_discrete: boolean, specify the action representation,
  * either 'discrete' or 'parametric'
  * True means discrete using  'String' as action,
  * False means parametric, using Map<BIGINT, DOUBLE>'
  *
  * add_terminal_state_row: boolean, if True assumes the final row
  * in each MDP corresponds to the terminal state and keeps it in output
  *
  * Columns of input table should contain:
  * mdp_id ( STRING ). A unique ID for the MDP chain that
  * this training example is a part of.
  *
  * state_features ( MAP<BIGINT,DOUBLE> ). The features of the current step
  * that are independent on the action.
  *
  * action ( STRING OR MAP<BIGINT,DOUBLE> ). The action taken at the current step.
  * A string if the action is discrete or
  * a set of features if the action is parametric.
  *
  * action_probability (DOUBLE). The probability that this action was taken.
  *
  * reward ( DOUBLE ). The reward at the current step.
  *
  * sequence_number ( BIGINT ).
  * A number representing the location of the state in the MDP.
  * There should be at most one row with the same mdp_id + sequence_number
  * (mdp_id + sequence_number makes a unique key).
  *
  * possible_actions ( ARRAY<STRING> OR ARRAY<MAP<BIGINT,DOUBLE>> ).
  * A list of actions that were possible at the current step.
  * This is optional but enables Q-Learning and improves model accuracy.
  *
  * metrics ( MAP<STRING, DOUBLE> )
  * The features used to calculate the reward at the current step
  *
  * This operator will generate output table with the following columns:
  * mdp_id ( STRING ). A unique ID for the MDP chain that
  * this training example is a part of.
  *
  * state_features ( MAP<BIGINT,DOUBLE> ). The features of the current step
  * that are independent on the action.
  *
  * action ( STRING OR MAP<BIGINT,DOUBLE> ). The action taken at the current step.
  * A string if the action is discrete or
  * a set of features if the action is parametric.
  *
  * action_probability (DOUBLE). The probability that this action was taken.
  *
  * reward ( ARRAY<DOUBLE> ). The rewards of consecutive n steps, starting at the current step.
  *
  * next_state_features ( ARRAY<MAP<BIGINT,DOUBLE>> ). The features of the subsequent n
  * steps that are action-independent.
  *
  * next_action (ARRAY<STRING> OR ARRAY<MAP<BIGINT, DOUBLE>> ). The action taken at
  * each of the next n steps
  *
  * sequence_number ( BIGINT ).
  * A number representing the location of the state in the MDP before
  * the sequence_number was converted to an ordinal number.
  * There should be at most one row with the same mdp_id +
  * sequence_number (mdp_id + sequence_number makes a unique key).
  *
  * sequence_number_ordinal ( BIGINT ).
  * A number representing the location of the state in the MDP.
  * There should be at most one row with the same mdp_id +
  * sequence_number_ordinal (mdp_id + sequence_number_ordinal makes
  * a unique key).
  *
  * time_diff ( ARRAY<BIGINT> ).
  * A list of numbers each representing the number of states between the current
  * state and one of the next n state. If the input table is sub-sampled
  * states will be missing. This column allows us to know how many
  * states are missing which can be used to adjust the discount factor.
  *
  * possible_actions ( ARRAY<STRING> OR ARRAY<MAP<BIGINT,DOUBLE>> )
  * A list of actions that were possible at the current step.
  *
  * possible_next_actions ( ARRAY<ARRAY<STRING>> OR ARRAY<ARRAY<MAP<BIGINT,DOUBLE>>> )
  * A list of actions that were possible at each of the next n steps.
  *
  * metrics (ARRAY<MAP<STRING, DOUBLE>>)
  * The features that are used to calculate the reward at consecutive n steps,
  * starting at the current step
  */
object MultiStepTimeline {

  private val log = LoggerFactory.getLogger(this.getClass.getName)
  def run(sqlContext: SQLContext, config: MultiStepTimelineConfiguration): Unit = {
    var terminalJoin = "";
    if (config.addTerminalStateRow) {
      terminalJoin = "LEFT OUTER";
    }
    var sortActionMethod = "UDF_SORT_ID";
    var sortPossibleActionMethod = "UDF_SORT_ARRAY_ID";
    if (!config.actionDiscrete) {
      sortActionMethod = "UDF_SORT_MAP";
      sortPossibleActionMethod = "UDF_SORT_ARRAY_MAP";
    }

    Helper.validateOrDestroyTrainingTable(sqlContext,
                                          config.outputTableName,
                                          config.actionDiscrete,
                                          false)
    MultiStepTimeline.createTrainingTable(sqlContext, config.outputTableName, config.actionDiscrete)
    MultiStepTimeline.registerUDFs(sqlContext)

    val sqlCommand = s"""
      WITH deduped as (
          SELECT
              mdp_id as mdp_id,
              FIRST(state_features) as state_features,
              FIRST(action) as action,
              FIRST(action_probability) as action_probability,
              FIRST(reward) as reward,
              FIRST(possible_actions) as possible_actions,
              FIRST(metrics) as metrics,
              FIRST(ds) as ds,
              sequence_number as sequence_number
              FROM (
                  SELECT * FROM ${config.inputTableName}
                  WHERE ds BETWEEN '${config.startDs}' AND '${config.endDs}'
              ) dummy
              GROUP BY mdp_id, sequence_number
      ),
      ordinal as (
          SELECT
              mdp_id as mdp_id,
              state_features as state_features,
              action as action,
              action_probability as action_probability,
              reward as reward,
              possible_actions as possible_actions,
              metrics as metrics,
              ds as ds,
              sequence_number as sequence_number,
              row_number() over (partition by mdp_id order by mdp_id, sequence_number) as sequence_number_ordinal,
              sequence_number - FIRST(sequence_number) OVER (
                  PARTITION BY mdp_id ORDER BY mdp_id, sequence_number
              ) AS time_since_first
              FROM deduped
      ),
      ordinal_join AS (
          SELECT
              first_sa.mdp_id AS mdp_id,
              first_sa.state_features AS state_features,
              first_sa.action AS action,
              first_sa.action_probability as action_probability,
              first_sa.reward AS reward,
              second_sa.reward AS next_reward,
              second_sa.state_features AS next_state_features,
              second_sa.action AS next_action,
              first_sa.sequence_number AS sequence_number,
              first_sa.sequence_number_ordinal AS sequence_number_ordinal,
              COALESCE(
                CAST(second_sa.sequence_number - first_sa.sequence_number AS BIGINT),
                first_sa.sequence_number
              ) AS time_diff,
              first_sa.time_since_first AS time_since_first,
              first_sa.possible_actions AS possible_actions,
              second_sa.possible_actions AS possible_next_actions,
              first_sa.metrics AS metrics,
              second_sa.metrics AS next_metrics
              FROM
                  ordinal first_sa
                  ${terminalJoin} JOIN ordinal second_sa
                  ON first_sa.mdp_id = second_sa.mdp_id
                  AND (first_sa.sequence_number_ordinal + 1) <= second_sa.sequence_number_ordinal
                  AND (first_sa.sequence_number_ordinal + ${config.steps}) >= second_sa.sequence_number_ordinal
      ),
      ordinal_join_time_diff AS (
          SELECT
              mdp_id AS mdp_id,
              state_features AS state_features,
              action AS action,
              action_probability as action_probability,
              reward AS reward,
              MAP(time_diff, next_reward) AS next_reward,
              MAP(time_diff, next_state_features) AS next_state_features,
              MAP(time_diff, next_action) AS next_action,
              sequence_number AS sequence_number,
              sequence_number_ordinal AS sequence_number_ordinal,
              time_diff AS time_diff,
              time_since_first AS time_since_first,
              possible_actions AS possible_actions,
              MAP(time_diff, possible_next_actions) AS possible_next_actions,
              metrics AS metrics,
              MAP(time_diff, next_metrics) AS next_metrics
              FROM
                  ordinal_join
      ),
      sarsa_unshuffled_multi_step AS (
          SELECT
              mdp_id AS mdp_id,
              FIRST(state_features) AS state_features,
              FIRST(action) AS action,
              FIRST(action_probability) as action_probability,
              UDF_PREPEND_DOUBLE(
                FIRST(reward),
                UDF_DROP_LAST_DOUBLE(UDF_SORT_DOUBLE(COLLECT_LIST(next_reward)))
              ) AS reward,
              UDF_SORT_MAP(
                COLLECT_LIST(next_state_features)
              ) AS next_state_features,
              ${sortActionMethod}(
                COLLECT_LIST(next_action)
              ) AS next_action,
              sequence_number AS sequence_number,
              sequence_number_ordinal AS sequence_number_ordinal,
              SORT_ARRAY(
                COLLECT_LIST(time_diff)
              ) AS time_diff,
              FIRST(time_since_first) AS time_since_first,
              FIRST(possible_actions) AS possible_actions,
              ${sortPossibleActionMethod}(
                COLLECT_LIST(possible_next_actions)
              ) AS possible_next_actions,
              UDF_PREPEND_MAP_STRING(
                FIRST(metrics),
                UDF_DROP_LAST_MAP_STRING(UDF_SORT_MAP_STRING(COLLECT_LIST(next_metrics)))
              ) AS metrics
          FROM
              ordinal_join_time_diff
          GROUP BY mdp_id, sequence_number, sequence_number_ordinal
      )
      INSERT OVERWRITE TABLE ${config.outputTableName} PARTITION(ds='${config.endDs}')
      SELECT
          mdp_id,
          state_features,
          action,
          action_probability,
          reward,
          next_state_features,
          next_action,
          sequence_number,
          sequence_number_ordinal,
          time_diff,
          time_since_first,
          possible_actions,
          possible_next_actions,
          metrics
      FROM
          sarsa_unshuffled_multi_step
          CLUSTER BY HASH(mdp_id, sequence_number)
    """.stripMargin
    log.info("Executing query: ")
    log.info(sqlCommand)
    sqlContext.sql(sqlCommand)
  }

  def createTrainingTable(sqlContext: SQLContext,
                          tableName: String,
                          actionDiscrete: Boolean): Unit = {
    var actionType = "STRING";
    var possibleActionType = "ARRAY<STRING>";
    if (!actionDiscrete) {
      actionType = "MAP<BIGINT, DOUBLE>"
      possibleActionType = "ARRAY<MAP<BIGINT,DOUBLE>>"
    }

    val sqlCommand = s"""
      CREATE TABLE IF NOT EXISTS ${tableName} (
          mdp_id STRING,
          state_features MAP <BIGINT, DOUBLE>,
          action ${actionType},
          action_probability DOUBLE,
          reward ARRAY<DOUBLE>,
          next_state_features ARRAY<MAP<BIGINT,DOUBLE>>,
          next_action ARRAY<${actionType}>,
          sequence_number BIGINT,
          sequence_number_ordinal BIGINT,
          time_diff ARRAY<BIGINT>,
          time_since_first BIGINT,
          possible_actions ${possibleActionType},
          possible_next_actions ARRAY<${possibleActionType}>,
          metrics ARRAY<MAP<STRING, DOUBLE>>
      ) PARTITIONED BY (ds STRING) TBLPROPERTIES ('RETENTION'='30')
    """.stripMargin
    log.info("Create query: ")
    log.info(sqlCommand)
    sqlContext.sql(sqlCommand);
  }

  def registerUDFs(sqlContext: SQLContext): Unit = {
    sqlContext.udf.register("UDF_PREPEND_DOUBLE", Udfs.prepend[Double] _)
    sqlContext.udf.register("UDF_PREPEND_MAP", Udfs.prepend[Map[Long, Double]] _)
    sqlContext.udf.register("UDF_PREPEND_MAP_STRING", Udfs.prepend[Map[String, Double]] _)

    sqlContext.udf.register("UDF_SORT_DOUBLE", Udfs.sort_list_of_map[Double] _)
    sqlContext.udf.register("UDF_SORT_ID", Udfs.sort_list_of_map[String] _)
    sqlContext.udf.register("UDF_SORT_MAP", Udfs.sort_list_of_map[Map[Long, Double]] _)
    sqlContext.udf.register("UDF_SORT_MAP_STRING", Udfs.sort_list_of_map[Map[String, Double]] _)
    sqlContext.udf.register("UDF_SORT_ARRAY_ID", Udfs.sort_list_of_map[Seq[String]] _)
    sqlContext.udf.register("UDF_SORT_ARRAY_MAP", Udfs.sort_list_of_map[Seq[Map[Long, Double]]] _)

    sqlContext.udf.register("UDF_DROP_LAST_DOUBLE", Udfs.drop_last[Double] _)
    sqlContext.udf.register("UDF_DROP_LAST_MAP", Udfs.drop_last[Map[Long, Double]] _)
    sqlContext.udf.register("UDF_DROP_LAST_MAP_STRING", Udfs.drop_last[Map[String, Double]] _)
  }
}
