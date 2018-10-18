// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import java.io.File
import org.scalactic.TolerantNumerics
import org.scalatest.Assertions._

import com.facebook.spark.common.testutil.PipelineTester

class TimelineTest extends PipelineTester {

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(0.01)

  test("two-state-mdp") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration("2018-01-01",
                                       "2018-01-01",
                                       false,
                                       true,
                                       "some_rl_input",
                                       "some_rl_timeline")

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          ("2018-01-01",
           "mdp1",
           1,
           1.0,
           "action1",
           0.8,
           Map("s1" -> 1.0),
           List("action1", "action2")), // First state
          ("2018-01-01", "mdp1", 11, 0.2, "action2", 0.7, Map("s2" -> 1.0), List()) // Second state
        ))
      .toDF("ds",
            "mdp_id",
            "sequence_number",
            "reward",
            "action",
            "action_probability",
            "state_features",
            "possible_actions")
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Create a mis-specified output table that will be deleted
    val bad_output = sparkContext
      .parallelize(
        List(
          ("2018-01-01"), // First state
          ("2018-01-01") // Second state
        ))
      .toDF("ds")
    bad_output.createOrReplaceTempView(config.outputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(Timeline.outputTableIsValid(sqlContext, s"${config.outputTableName}"))

    // Query the results
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show()
    assert(df.count() == 1)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[String, Double]](2) == Map("s1" -> 1.0))
    assert(firstRow.getAs[String](3) == "action1")
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Double](5) === 1.0)
    assert(firstRow.getAs[Map[String, Double]](6) == Map("s2" -> 1.0))
    assert(firstRow.getAs[String](7) == "action2")
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Long](10) == 10)
    assert(firstRow.getAs[Seq[String]](11) == List("action1", "action2"))
    assert(firstRow.getAs[Seq[String]](12) == List())
    assert(firstRow.getAs[Map[Long, Double]](13) == Map(0 -> 1.0, 10 -> 0.2))
    assert(firstRow.getAs[Map[Long, Double]](14) == Map(0 -> 1.0, 1 -> 0.2))
  }
}
