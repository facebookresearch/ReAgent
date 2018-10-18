// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class TimelineConfiguration(startDs: String,
                                 endDs: String,
                                 addTerminalStateRow: Boolean,
                                 actionDiscrete: Boolean,
                                 inputTableName: String,
                                 outputTableName: String,
                                 numOutputShards: Int = 1)

/**
  * Given table of state, action, mdp_id, sequence_number, reward, possible_next_actions
  * return the table needed for reinforcement learning (MDP: Markov Decision Process)
  * mdp_id, state_features, action, reward, next_state_features, next_action,
  * sequence_number, sequence_number_ordinal, time_diff, possible_next_actions,
  * reward_timeline. Shuffles the results.
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
  * False means parametric, using Map<String, DOUBLE>'
  *
  * add_terminal_state_row: boolean, if True assumes the final row
  * in each MDP corresponds to the terminal state and keeps it in output
  *
  * Columns of input table should contain:
  * mdp_id ( STRING ). A unique ID for the MDP chain that
  * this training example is a part of.
  *
  * state_features ( MAP<STRING,DOUBLE> ). The features of the current step
  * that are independent on the action.
  *
  * action ( STRING OR MAP<STRING,DOUBLE> ). The action taken at the current step.
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
  * possible_actions ( ARRAY<STRING> OR ARRAY<MAP<STRING,DOUBLE>> ).
  * A list of actions that were possible at the current step.
  * This is optional but enables Q-Learning and improves model accuracy.
  *
  * This operator will generate output table with the following columns:
  * mdp_id ( STRING ). A unique ID for the MDP chain that
  * this training example is a part of.
  *
  * state_features ( MAP<STRING,DOUBLE> ). The features of the current step
  * that are independent on the action.
  *
  * action ( STRING OR MAP<STRING,DOUBLE> ). The action taken at the current step.
  * A string if the action is discrete or
  * a set of features if the action is parametric.
  *
  * action_probability (DOUBLE). The probability that this action was taken.
  *
  * reward ( DOUBLE ). The reward at the current step.
  *
  * next_state_features ( MAP<STRING,DOUBLE> ). The features of the subsequent
  * step that are action-independent.
  *
  * next_action (STRING OR MAP<STRING, DOUBLE> ). The action taken at the next step
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
  * time_diff ( BIGINT ).
  * A number representing the number of states between the current
  * state and next state. If the input table is sub-sampled
  * states will be missing. This column allows us to know how many
  * states are missing which can be used to adjust the discount factor.
  *
  * possible_actions ( ARRAY<STRING> OR ARRAY<MAP<STRING,DOUBLE>> )
  * A list of actions that were possible at the current step.
  *
  * possible_next_actions ( ARRAY<STRING> OR ARRAY<MAP<STRING,DOUBLE>> )
  * A list of actions that were possible at the next step.
  *
  * reward_timeline ( MAP<INT, DOUBLE> ). A map containing the future reward.
  * Each key is the number of timesteps forward,
  * and the value is the reward at that timestep.
  * The key with index 0 should have a value equal to the reward column.
  * This column is optional but used to measure the model performance.
  *
  */
object Timeline {

  private val log = LoggerFactory.getLogger(this.getClass.getName)
  def run(sqlContext: SQLContext, config: TimelineConfiguration): Unit = {
    var terminalJoin = "";
    if (config.addTerminalStateRow) {
      terminalJoin = "LEFT OUTER";
    }

    Timeline.validateOrDestroyTrainingTable(sqlContext,
                                            config.outputTableName,
                                            config.actionDiscrete)
    Timeline.createTrainingTable(sqlContext, config.outputTableName, config.actionDiscrete)

    sqlContext.udf.register("UNION_LIST_OF_REWARD_TIMELINES", Udfs.unionListOfMaps[Long, Double] _)

    val sqlCommand = s"""
      WITH deduped as (
          SELECT
              mdp_id as mdp_id,
              FIRST(state_features) as state_features,
              FIRST(action) as action,
              FIRST(action_probability) as action_probability,
              FIRST(reward) as reward,
              FIRST(possible_actions) as possible_actions,
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
              ds as ds,
              sequence_number as sequence_number,
              row_number() over (partition by mdp_id order by mdp_id, sequence_number) as sequence_number_ordinal
              FROM deduped
      ),
      reward_only AS (
          SELECT
              mdp_id,
              sequence_number,
              sequence_number_ordinal,
              reward
          FROM
              ordinal
      ),
      reward_timeline AS (
          SELECT
              t1.mdp_id AS mdp_id,
              t1.sequence_number AS sequence_number,
              UNION_LIST_OF_REWARD_TIMELINES(
                COLLECT_LIST(
                  MAP(CAST((t2.sequence_number - t1.sequence_number) AS BIGINT), t2.reward))) AS reward_timeline,
              UNION_LIST_OF_REWARD_TIMELINES(
                COLLECT_LIST(
                  MAP(CAST((t2.sequence_number_ordinal - t1.sequence_number_ordinal) AS BIGINT), t2.reward))) AS reward_timeline_ordinal
          FROM
              reward_only t1
              JOIN reward_only t2
              ON t1.mdp_id = t2.mdp_id
          WHERE
              t1.sequence_number_ordinal <= t2.sequence_number_ordinal
              AND t1.sequence_number_ordinal + 100 > t2.sequence_number_ordinal -- Disregard rewards too far in the future
              AND (t2.reward != 0 OR t1.sequence_number_ordinal = t2.sequence_number_ordinal) -- Only include reward=0 for t=0
          GROUP BY
              t1.mdp_id,
              t1.sequence_number
      ),
      sarsa_unshuffled AS (
          SELECT
              first_sa.mdp_id AS mdp_id,
              first_sa.state_features AS state_features,
              first_sa.action AS action,
              first_sa.action_probability as action_probability,
              first_sa.reward AS reward,
              rt.reward_timeline AS reward_timeline,
              rt.reward_timeline_ordinal AS reward_timeline_ordinal,
              second_sa.state_features AS next_state_features,
              second_sa.action AS next_action,
              first_sa.sequence_number AS sequence_number,
              first_sa.sequence_number_ordinal AS sequence_number_ordinal,
              second_sa.sequence_number - first_sa.sequence_number AS time_diff,
              first_sa.possible_actions AS possible_actions,
              second_sa.possible_actions AS possible_next_actions
          FROM
              ordinal first_sa
              ${terminalJoin} JOIN ordinal second_sa
              ON first_sa.mdp_id = second_sa.mdp_id
              AND (first_sa.sequence_number_ordinal + 1) = second_sa.sequence_number_ordinal
              JOIN reward_timeline rt
              ON first_sa.mdp_id = rt.mdp_id
              AND first_sa.sequence_number = rt.sequence_number
      )
      INSERT OVERWRITE TABLE ${config.outputTableName} PARTITION(ds='${config.endDs}')
      SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
                          .slice(1, Constants.TRAINING_DATA_COLUMN_NAMES.length)
                          .mkString(",")}
      FROM
          sarsa_unshuffled
          DISTRIBUTE BY -1 - PMOD(HASH(mdp_id, sequence_number_ordinal), 10007)
          SORT BY -1 - PMOD(HASH(mdp_id, sequence_number_ordinal), 10007)
    """.stripMargin
    log.info("Executing query: ")
    log.info(sqlCommand)
    sqlContext.sql(sqlCommand)
  }

  def outputTableIsValid(sqlContext: SQLContext, tableName: String): Boolean = {
    val checkOutputTableCommand = s"""
    SELECT * FROM ${tableName} LIMIT 1
    """
    val checkOutputDf = sqlContext.sql(checkOutputTableCommand)
    return checkOutputDf.columns.size == Constants.TRAINING_DATA_COLUMN_NAMES.length
  }

  def validateOrDestroyTrainingTable(sqlContext: SQLContext,
                                     tableName: String,
                                     actionDiscrete: Boolean): Unit = {
    var actionType = "STRING";
    var possibleActionType = "ARRAY<STRING>";
    if (!actionDiscrete) {
      actionType = "MAP<STRING, DOUBLE>"
      possibleActionType = "ARRAY<MAP<STRING,DOUBLE>>"
    }

    try {
      val checkOutputTableCommand = s"""
      DESCRIBE ${tableName}
      """
      val df = sqlContext.sql(checkOutputTableCommand);
      // Validate the schema and destroy the output table if it doesn't match
      var validTable = Timeline.outputTableIsValid(sqlContext, tableName)
      if (!validTable) {
        val dropTableCommand = s"""
        DROP TABLE ${tableName}
        """
        sqlContext.sql(dropTableCommand);
      }
    } catch {
      case e: org.apache.spark.sql.catalyst.analysis.NoSuchTableException => {}
      case e: Throwable                                                   => log.error(e.toString())
    }
  }

  def createTrainingTable(sqlContext: SQLContext,
                          tableName: String,
                          actionDiscrete: Boolean): Unit = {
    var actionType = "STRING";
    var possibleActionType = "ARRAY<STRING>";
    if (!actionDiscrete) {
      actionType = "MAP<STRING, DOUBLE>"
      possibleActionType = "ARRAY<MAP<STRING,DOUBLE>>"
    }

    val sqlCommand = s"""
CREATE TABLE IF NOT EXISTS ${tableName} (
    mdp_id STRING,
    state_features MAP < STRING,
    DOUBLE >,
    action ${actionType},
    action_probability DOUBLE,
    reward DOUBLE,
    next_state_features MAP < STRING,
    DOUBLE >,
    next_action ${actionType},
    sequence_number BIGINT,
    sequence_number_ordinal BIGINT,
    time_diff BIGINT,
    possible_actions ${possibleActionType},
    possible_next_actions ${possibleActionType},
    reward_timeline MAP < BIGINT, DOUBLE >,
    reward_timeline_ordinal MAP < BIGINT, DOUBLE >
) PARTITIONED BY (ds STRING) TBLPROPERTIES ('RETENTION'='30')
""".stripMargin
    sqlContext.sql(sqlCommand);
  }
}
