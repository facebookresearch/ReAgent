// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._

object Helper {

  private val log = LoggerFactory.getLogger(this.getClass.getName)

  def next_step_col_name(col_name: String): String =
    if (col_name == "possible_actions") "possible_next_actions" else "next_" + col_name

  def next_step_col_type(col_type: String, next_step_col_is_arr: Boolean): String =
    if (next_step_col_is_arr) s"array<${col_type}>" else col_type

  def outputTableIsValid(
      sqlContext: SQLContext,
      tableName: String,
      actionDataType: String = "string",
      rewardTypes: Map[String, String] = Constants.DEFAULT_REWARD_TYPES,
      timelineJoinTypes: Map[String, String] = Map("possible_actions" -> "array<string>"),
      next_step_col_is_arr: Boolean = false
  ): Boolean = {
    // check column types
    val dt = sqlContext.sparkSession.catalog
      .listColumns(tableName)
      .collect
      .map(column => column.name -> column.dataType)
      .toMap

    val nextActionDataType = this.next_step_col_type(actionDataType, next_step_col_is_arr)
    (
      actionDataType == dt.getOrElse("action", "") &&
      nextActionDataType == dt.getOrElse("next_action", "") &&
      rewardTypes.filter { case (k, v) => (v == dt.getOrElse(k, "")) }.size == rewardTypes.size &&
      timelineJoinTypes.filter {
        case (k, v) =>
          (v == dt.getOrElse(k, "") &&
            this.next_step_col_type(v, next_step_col_is_arr) == dt.getOrElse(
              this.next_step_col_name(k),
              ""
            ))
      }.size == timelineJoinTypes.size
    )
  }

  def getDataTypes(
      sqlContext: SQLContext,
      tableName: String,
      columnNames: List[String]
  ): Map[String, String] = {
    val dt = sqlContext.sparkSession.catalog
      .listColumns(tableName)
      .collect
      .filter(column => columnNames.contains(column.name))
      .map(column => column.name -> column.dataType)
      .toMap
    assert(dt.size == columnNames.size)
    dt
  }

  def destroyTrainingTable(
      sqlContext: SQLContext,
      tableName: String
  ): Unit =
    try {
      val dropTableCommand = s"""
      DROP TABLE ${tableName}
      """
      sqlContext.sql(dropTableCommand);
    } catch {
      case e: org.apache.spark.sql.catalyst.analysis.NoSuchTableException => {}
      case e: Throwable                                                   => log.error(e.toString())
    }

}
