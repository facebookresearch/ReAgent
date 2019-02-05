// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.apache.spark.sql._

object Helper {

  def outputTableIsValid(sqlContext: SQLContext,
                         tableName: String,
                         actionDiscrete: Boolean): Boolean = {
    // check number of columns
    val checkOutputTableCommand = s"""
     SELECT * FROM ${tableName} LIMIT 1
     """
    val checkOutputDf = sqlContext.sql(checkOutputTableCommand)
    if (checkOutputDf.columns.size != Constants.TRAINING_DATA_COLUMN_NAMES.length)
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
    else
      return true
  }

}
