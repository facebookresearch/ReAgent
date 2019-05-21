// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._

object Helper {

  private val log = LoggerFactory.getLogger(this.getClass.getName)

  def outputTableIsValid(sqlContext: SQLContext,
                         tableName: String,
                         actionDiscrete: Boolean,
                         extraFeatureColumnTypes: Map[String, String] = Map()): Boolean = {
    val totalColumns = Constants.TRAINING_DATA_COLUMN_NAMES.size + 2 * extraFeatureColumnTypes.size

    // check column types
    var actionType = "string";
    var possibleActionType = "array<string>";
    if (!actionDiscrete) {
      actionType = "map<bigint,double>"
      possibleActionType = "array<map<bigint,double>>"
    }
    val dt = sqlContext.sparkSession.catalog
      .listColumns(tableName)
      .collect
      .map(column => column.name -> column.dataType)
      .toMap

    (
      dt.size == totalColumns &&
      actionType == dt.getOrElse("action", "") &&
      possibleActionType == dt.getOrElse("possible_actions", "") &&
      extraFeatureColumnTypes.filter {
        case (k, v) => (v == dt.getOrElse(k, "") && v == dt.getOrElse(s"next_${k}", ""))
      }.size == extraFeatureColumnTypes.size
    )
  }

  def getDataTypes(sqlContext: SQLContext,
                   tableName: String,
                   columnNames: List[String]): Map[String, String] = {
    val dt = sqlContext.sparkSession.catalog
      .listColumns(tableName)
      .collect
      .filter(column => columnNames.contains(column.name))
      .map(column => column.name -> column.dataType)
      .toMap
    assert(dt.size == columnNames.size)
    dt
  }

  def validateOrDestroyTrainingTable(sqlContext: SQLContext,
                                     tableName: String,
                                     actionDiscrete: Boolean,
                                     extraFeatureTypes: Map[String, String] = Map()): Unit =
    try {
      // Validate the schema and destroy the output table if it doesn't match
      var validTable = Helper.outputTableIsValid(
        sqlContext,
        tableName,
        actionDiscrete,
        extraFeatureTypes
      )
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
