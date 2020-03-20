// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import scala.math.abs

import org.slf4j.LoggerFactory
import org.apache.spark.sql._
import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

case class ExtraFeatureColumn(columnName: String, columnType: String)

case class TimelineConfiguration(
    startDs: String,
    endDs: String,
    addTerminalStateRow: Boolean,
    actionDiscrete: Boolean,
    inputTableName: String,
    outputTableName: String,
    evalTableName: String,
    numOutputShards: Int,
    includePossibleActions: Boolean = true,
    outlierEpisodeLengthPercentile: Option[Double] = None,
    percentileFunction: String = "percentile_approx",
    rewardColumns: List[String] = Constants.DEFAULT_REWARD_COLUMNS,
    extraFeatureColumns: List[String] = Constants.DEFAULT_EXTRA_FEATURE_COLUMNS,
    timeWindowLimit: Option[Long] = None,
    validationSql: Option[String] = None
)

/**
  * Given table of state, action, mdp_id, sequence_number, reward, possible_next_actions
  * return the table needed for reinforcement learning (MDP: Markov Decision Process)
  * mdp_id, state_features, action, reward, next_state_features, next_action,
  * sequence_number, sequence_number_ordinal, time_diff, possible_next_actions.
  * Shuffles the results.
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
  * reward ( DOUBLE ). The reward at the current step.
  *
  * next_state_features ( MAP<BIGINT,DOUBLE> ). The features of the subsequent
  * step that are action-independent.
  *
  * next_action (STRING OR MAP<BIGINT, DOUBLE> ). The action taken at the next step
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
  * possible_actions ( ARRAY<STRING> OR ARRAY<MAP<BIGINT,DOUBLE>> )
  * A list of actions that were possible at the current step.
  *
  * possible_next_actions ( ARRAY<STRING> OR ARRAY<MAP<BIGINT,DOUBLE>> )
  * A list of actions that were possible at the next step.
  *
  * config.validationSql (Option[String], default None).
  * A SQL query to validate against a Timeline Pipeline input/output table where
  * result should have only one row and that row contains only true booleans
  * Ex: select if((select count(*) from {config.outputTableName} where mdp_id<0) == 0, TRUE, FALSE)
  * Ex: select if((select count(*) from {config.inputTableName} where reward>1.0) == 0, TRUE, FALSE)
  */
object Timeline {

  private val log = LoggerFactory.getLogger(this.getClass.getName)
  def run(
      sqlContext: SQLContext,
      config: TimelineConfiguration
  ): Unit = {
    var filterTerminal = "HAVING next_state_features IS NOT NULL";
    if (config.addTerminalStateRow) {
      filterTerminal = "";
    }

    val actionDataType =
      Helper.getDataTypes(sqlContext, config.inputTableName, List("action"))("action")
    log.info("action column data type:" + s"${actionDataType}")

    var timelineJoinColumns = config.extraFeatureColumns
    if (config.includePossibleActions) {
      timelineJoinColumns = "possible_actions" :: timelineJoinColumns
    }

    val rewardColumnDataTypes =
      Helper.getDataTypes(sqlContext, config.inputTableName, config.rewardColumns)
    log.info("reward columns:" + s"${config.rewardColumns}")
    log.info("reward column types:" + s"${rewardColumnDataTypes}")

    val timelineJoinColumnDataTypes =
      Helper.getDataTypes(sqlContext, config.inputTableName, timelineJoinColumns)
    log.info("timeline join column columns:" + s"${timelineJoinColumns}")
    log.info("timeline join column types:" + s"${timelineJoinColumnDataTypes}")

    Timeline.createTrainingTable(
      sqlContext,
      config.outputTableName,
      actionDataType,
      rewardColumnDataTypes,
      timelineJoinColumnDataTypes
    )

    config.outlierEpisodeLengthPercentile.foreach { percentile =>
      sqlContext.sql(s"""
          SELECT mdp_id, COUNT(mdp_id) AS mdp_length
          FROM ${config.inputTableName}
          WHERE ds BETWEEN '${config.startDs}' AND '${config.endDs}'
          GROUP BY mdp_id
      """).createOrReplaceTempView("episode_length")
    }

    val lengthThreshold = Timeline.mdpLengthThreshold(sqlContext, config)

    val mdpFilter = lengthThreshold
      .map { threshold =>
        s"mdp_filter AS (SELECT mdp_id FROM episode_length WHERE mdp_length <= ${threshold}),"
      }
      .getOrElse("")

    val joinClause = lengthThreshold
      .map { threshold =>
        s"""
        JOIN mdp_filter
        WHERE a.mdp_id = mdp_filter.mdp_id AND
    """.stripMargin
      }
      .getOrElse("WHERE")

    val rewardSourceColumns = rewardColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) => s"${acc}, a.${k}"
    }
    val timelineSourceColumns = timelineJoinColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) => s"${acc}, a.${k}"
    }

    val timeLimitedSourceTable = config.timeWindowLimit
      .map { timeLimit =>
        s"""
        , time_limited_source_table AS (
            SELECT
                *,
                sequence_number - FIRST(sequence_number) OVER (
                     PARTITION BY mdp_id
                     ORDER BY mdp_id, sequence_number
                ) AS time_since_first
            FROM source_table
            HAVING time_since_first <= ${timeLimit}
        )
        """.stripMargin
      }
      .getOrElse("")

    val sourceTable = s"""
    WITH ${mdpFilter}
        source_table AS (
            SELECT
                a.mdp_id,
                a.state_features,
                a.action_probability,
                a.action
                ${rewardSourceColumns},
                a.sequence_number
                ${timelineSourceColumns}
            FROM ${config.inputTableName} a
            ${joinClause}
            a.ds BETWEEN '${config.startDs}' AND '${config.endDs}'
        )
        ${timeLimitedSourceTable}
    """.stripMargin

    val sourceTableName = config.timeWindowLimit
      .map { _ =>
        "time_limited_source_table"
      }
      .getOrElse("source_table")

    val rewardColumnsQuery = rewardColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) => s"${acc}, ${k}"
    }
    val timelineJoinColumnsQuery = timelineJoinColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) =>
        s"""
        ${acc},
        ${k},
        LEAD(${k}) OVER (
            PARTITION BY
                mdp_id
              ORDER BY
                  mdp_id,
                  sequence_number
          ) AS ${Helper.next_step_col_name(k)}
        """
    }

    val sqlCommand = s"""
    ${sourceTable}
    SELECT
        mdp_id,
        state_features,
        action,
        LEAD(action) OVER (
            PARTITION BY
                mdp_id
            ORDER BY
                mdp_id,
                sequence_number
        ) AS next_action,
        action_probability
        ${rewardColumnsQuery},
        LEAD(state_features) OVER (
            PARTITION BY
                mdp_id
            ORDER BY
                mdp_id,
                sequence_number
        ) AS next_state_features,
        sequence_number,
        ROW_NUMBER() OVER (
            PARTITION BY
                mdp_id
            ORDER BY
                mdp_id,
                sequence_number
        ) AS sequence_number_ordinal,
        COALESCE(LEAD(sequence_number) OVER (
            PARTITION BY
                mdp_id
            ORDER BY
                mdp_id,
                sequence_number
        ), sequence_number) - sequence_number AS time_diff,
        sequence_number - FIRST(sequence_number) OVER (
            PARTITION BY
                mdp_id
            ORDER BY
                mdp_id,
                sequence_number
        ) AS time_since_first
        ${timelineJoinColumnsQuery}
    FROM ${sourceTableName}
    ${filterTerminal}
    CLUSTER BY HASH(mdp_id, sequence_number)
    """.stripMargin
    log.info("Executing query: ")
    log.info(sqlCommand)
    var df = sqlContext.sql(sqlCommand)
    log.info("Done with query")

    // Handle nulls in output present when terminal states are present
    val handle_cols = timelineJoinColumnDataTypes.++(
      Map(
        "action" -> actionDataType,
        "state_features" -> "map<bigint,double>"
      )
    )
    for ((col_name, col_type) <- handle_cols) {
      val next_col_name = Helper.next_step_col_name(col_name)
      val empty_placeholder = col_type match {
        case "string"                                => Udfs.emptyStr()
        case "array<string>"                         => Udfs.emptyArrOfStr()
        case "map<bigint,double>"                    => Udfs.emptyMap()
        case "array<map<bigint,double>>"             => Udfs.emptyArrOfMap()
        case "array<bigint>"                         => Udfs.emptyArrOfLong()
        case "map<bigint,array<bigint>>"             => Udfs.emptyMapOfIds()
        case "map<bigint,map<bigint,double>>"        => Udfs.emptyMapOfMap()
        case "map<bigint,array<map<bigint,double>>>" => Udfs.emptyMapOfArrOfMap()
      }
      df = df
        .withColumn(next_col_name, coalesce(df(next_col_name), empty_placeholder))
    }

    val stagingTable = "stagingTable_" + config.outputTableName
    if (sqlContext.tableNames.contains(stagingTable)) {
      log.warn("RL ValidationSql staging table name collision occurred, name: " + stagingTable)
    }
    df.createOrReplaceTempView(stagingTable)

    val maybeError = config.validationSql.flatMap { query =>
      Helper.validateTimeline(
        sqlContext,
        query
          .replace("{config.outputTableName}", stagingTable)
          .replace("{config.inputTableName}", config.inputTableName)
      )
    }

    assert(maybeError.isEmpty, "validationSql validation failure: " + maybeError)

    val insertCommandOutput = s"""
      INSERT OVERWRITE TABLE ${config.outputTableName} PARTITION(ds='${config.endDs}')
      SELECT * FROM ${stagingTable}
    """.stripMargin
    sqlContext.sql(insertCommandOutput)
  }

  def mdpLengthThreshold(sqlContext: SQLContext, config: TimelineConfiguration): Option[Double] =
    config.outlierEpisodeLengthPercentile.flatMap { percentile =>
      {
        val df = sqlContext.sql(s"""
            WITH a AS (
                SELECT ${config.percentileFunction}(mdp_length, ${percentile}) pct FROM episode_length
            ),
            b AS (
                SELECT
                    count(*) as mdp_count,
                    sum(IF(episode_length.mdp_length > a.pct, 1, 0)) as outlier_count
                FROM episode_length CROSS JOIN a
            )
            SELECT a.pct, b.mdp_count, b.outlier_count
            FROM b CROSS JOIN a
        """)
        val res = df.first
        // episode length at x percentile could be either Long or Double,
        // depending on percentileFunction being used
        val pct_val = res.schema("pct").dataType match {
          case DoubleType => res.getAs[Double]("pct")
          case LongType   => res.getAs[Long]("pct")
        }
        val mdp_count = res.getAs[Long]("mdp_count")
        val outlier_count = res.getAs[Long]("outlier_count")
        log.info(s"Threshold: ${pct_val}; mdp count: ${mdp_count}; outlier_count: ${outlier_count}")
        val outlier_percent = outlier_count.toDouble / mdp_count
        val expected_outlier_percent = 1.0 - percentile
        if (abs(outlier_percent - expected_outlier_percent) / expected_outlier_percent > 0.1) {
          log.warn(
            s"Outlier percent mismatch; expected: ${expected_outlier_percent}; got ${outlier_percent}"
          )
          None
        } else
          Some(pct_val)
      }
    }

  def createTrainingTable(
      sqlContext: SQLContext,
      tableName: String,
      actionDataType: String,
      rewardColumnDataTypes: Map[String, String] = Map("reward" -> "double"),
      timelineJoinColumnDataTypes: Map[String, String] = Map()
  ): Unit = {
    val rewardColumns = rewardColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) => s"${acc}, ${k} ${v}"
    }
    val timelineJoinColumns = timelineJoinColumnDataTypes.foldLeft("") {
      case (acc, (k, v)) => s"${acc}, ${k} ${v}, ${Helper.next_step_col_name(k)} ${v}"
    }

    val sqlCommand = s"""
CREATE TABLE IF NOT EXISTS ${tableName} (
    mdp_id STRING,
    state_features MAP < BIGINT,
    DOUBLE >,
    action ${actionDataType},
    next_action ${actionDataType},
    action_probability DOUBLE
    ${rewardColumns},
    next_state_features MAP < BIGINT,
    DOUBLE >,
    sequence_number BIGINT,
    sequence_number_ordinal BIGINT,
    time_diff BIGINT,
    time_since_first BIGINT
    ${timelineJoinColumns}
) PARTITIONED BY (ds STRING) TBLPROPERTIES ('RETENTION'='30')
""".stripMargin
    sqlContext.sql(sqlCommand);
  }
}
