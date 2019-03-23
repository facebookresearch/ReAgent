// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._

object Helper {

  private val log = LoggerFactory.getLogger(this.getClass.getName)

  def outputTableIsValid(sqlContext: SQLContext,
                         tableName: String,
                         actionDiscrete: Boolean,
                         includeSparseData: Boolean): Boolean = {
    // check number of columns
    val checkOutputTableCommand = s"""
     SELECT * FROM ${tableName} LIMIT 1
     """
    val checkOutputDf = sqlContext.sql(checkOutputTableCommand)
    var totalColumns = Constants.TRAINING_DATA_COLUMN_NAMES
    if (includeSparseData) {
      totalColumns = totalColumns ++ Constants.SPARSE_DATA_COLUMN_NAMES
    }

    if (checkOutputDf.columns.size != totalColumns.length)
      return false

    // check column types
    var actionType = "string";
    var possibleActionType = "array<string>";
    if (!actionDiscrete) {
      actionType = "map<bigint,double>"
      possibleActionType = "array<map<bigint,double>>"
    }
    val describeTableCommand = s"""
      DESCRIBE ${tableName}
      """
    val describeTableDf = sqlContext.sql(describeTableCommand);
    if (describeTableDf.filter("col_name=='action'").count() != 1)
      return false
    else if (actionType != describeTableDf.filter("col_name=='action'").head.getAs[String](1))
      return false
    else if (describeTableDf.filter("col_name=='possible_actions'").count() != 1)
      return false
    else if (possibleActionType != describeTableDf
               .filter("col_name=='possible_actions'")
               .head
               .getAs[String](1))
      return false
    else if (includeSparseData && ("map<bigint,array<bigint>>" != describeTableDf
               .filter("col_name=='state_id_list_features'")
               .head
               .getAs[String](1)))
      return false
    else
      return true
  }

  def validateOrDestroyTrainingTable(sqlContext: SQLContext,
                                     tableName: String,
                                     actionDiscrete: Boolean,
                                     includeSparseData: Boolean): Unit =
    try {
      // Validate the schema and destroy the output table if it doesn't match
      var validTable = Helper.outputTableIsValid(
        sqlContext,
        tableName,
        actionDiscrete,
        includeSparseData
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
