// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import java.io.File
import org.scalactic.TolerantNumerics
import org.scalatest.Assertions._

import com.facebook.spark.common.testutil.PipelineTester

class TimelineTest extends PipelineTester {

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(0.01)

  test("two-state-discrete-mdp-one-step-rl") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = MultiStepTimelineConfiguration("2018-01-01",
                                                "2018-01-01",
                                                false,
                                                action_discrete,
                                                "some_rl_input",
                                                "some_rl_timeline",
                                                null,
                                                1,
                                                1)
    // destroy previous schema
    MultiStepTimeline.validateOrDestroyTrainingTable(sqlContext,
                                                     s"${config.outputTableName}",
                                                     action_discrete)

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
           List("action1", "action2"),
           Map("Widgets" -> 10.0)), // First state
          ("2018-01-01",
           "mdp1",
           11,
           0.2,
           "action2",
           0.7,
           Map("s2" -> 1.0),
           List(),
           Map("Widgets" -> 20.0)) // Second state
        ))
      .toDF("ds",
            "mdp_id",
            "sequence_number",
            "reward",
            "action",
            "action_probability",
            "state_features",
            "possible_actions",
            "metrics")
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
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(sqlContext, s"${config.outputTableName}", action_discrete)
    )

    // Query the results
    println("----------------Input table----------------");
    val sqlInputCommand = s"""
         SELECT * FROM ${config.inputTableName}
         WHERE ds BETWEEN '${config.startDs}' AND '${config.endDs}'
    """.stripMargin
    sqlCtx.sql(sqlInputCommand).show(false);
    println("----------------Output table----------------");
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show(false)
    println("----------------Output table schema----------------");
    df.printSchema()
    assert(df.count() == 1)

    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[String, Double]](2) == Map("s1" -> 1.0))
    assert(firstRow.getAs[String](3) == "action1")
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Seq[Double]](5) === List(1.0))
    assert(firstRow.getAs[Seq[Map[String, Double]]](6) == List(Map("s2" -> 1.0)))
    assert(firstRow.getAs[Seq[String]](7) == List("action2"))
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Seq[Long]](10) == List(10))
    assert(firstRow.getAs[Seq[String]](11) == List("action1", "action2"))
    // special case: possible_next_actions is a list (not list of list) when
    // possible_next_actions in all next n steps (here n = 1) are empty lists
    assert(firstRow.getAs[Seq[Seq[String]]](12) == List())
    assert(firstRow.getAs[Seq[Map[String, Double]]](13) == List(Map("Widgets" -> 10.0)))
  }

  test("four-state-discrete-mdp-three-step-rl") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = MultiStepTimelineConfiguration("2018-01-01",
                                                "2018-01-01",
                                                false,
                                                action_discrete,
                                                "some_rl_input",
                                                "some_rl_timeline",
                                                null,
                                                1,
                                                3)
    // destroy previous schema
    MultiStepTimeline.validateOrDestroyTrainingTable(sqlContext,
                                                     s"${config.outputTableName}",
                                                     action_discrete)

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
           List("action1", "action2"),
           Map("Widgets" -> 10.0)), // First state
          ("2018-01-01",
           "mdp1",
           11,
           0.2,
           "action2",
           0.7,
           Map("s2" -> 1.0),
           List("action2", "action3"),
           Map("Widgets" -> 20.0)), // Second state
          ("2018-01-01",
           "mdp1",
           12,
           0.9,
           "action3",
           0.6,
           Map("s3" -> 1.0),
           List("action3", "action4"),
           Map("Widgets" -> 30.0)), // Third state
          ("2018-01-01",
           "mdp1",
           13,
           0.4,
           "action4",
           0.5,
           Map("s4" -> 1.0),
           List("action4", "action1", "action2", "action3"),
           Map("Widgets" -> 40.0)) // Fourth state
        ))
      .toDF("ds",
            "mdp_id",
            "sequence_number",
            "reward",
            "action",
            "action_probability",
            "state_features",
            "possible_actions",
            "metrics")
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
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(sqlContext, s"${config.outputTableName}", action_discrete)
    )

    // Query the results
    println("----------------Input table----------------");
    val sqlInputCommand = s"""
         SELECT * FROM ${config.inputTableName}
         WHERE ds BETWEEN '${config.startDs}' AND '${config.endDs}'
    """.stripMargin
    sqlCtx.sql(sqlInputCommand).show(false);
    println("----------------Output table----------------");
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show(false)
    println("----------------Output table schema----------------");
    df.printSchema()
    assert(df.count() == 3)

    val firstRec = df.where($"sequence_number_ordinal" === 1).head()
    assert(firstRec.getAs[String](0) == "2018-01-01")
    assert(firstRec.getAs[String](1) == "mdp1")
    assert(firstRec.getAs[Map[String, Double]](2) == Map("s1" -> 1.0))
    assert(firstRec.getAs[String](3) == "action1")
    assert(firstRec.getAs[Double](4) == 0.8)
    assert(firstRec.getAs[Seq[Double]](5) == List(1.0, 0.2, 0.9))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](6) ==
        List(Map("s2" -> 1.0), Map("s3" -> 1.0), Map("s4" -> 1.0)))
    assert(firstRec.getAs[Seq[String]](7) == List("action2", "action3", "action4"))
    assert(firstRec.getAs[Long](8) == 1)
    assert(firstRec.getAs[Long](9) == 1)
    assert(firstRec.getAs[Seq[Long]](10) == List(10, 11, 12))
    assert(firstRec.getAs[Seq[String]](11) == List("action1", "action2"))
    assert(
      firstRec.getAs[Seq[Seq[String]]](12) ==
        List(List("action2", "action3"),
             List("action3", "action4"),
             List("action4", "action1", "action2", "action3")))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](13) ==
        List(Map("Widgets" -> 10.0), Map("Widgets" -> 20.0), Map("Widgets" -> 30.0)))

    val secondRec = df.where($"sequence_number_ordinal" === 2).head()
    assert(secondRec.getAs[String](0) == "2018-01-01")
    assert(secondRec.getAs[String](1) == "mdp1")
    assert(secondRec.getAs[Map[String, Double]](2) == Map("s2" -> 1.0))
    assert(secondRec.getAs[String](3) == "action2")
    assert(secondRec.getAs[Double](4) == 0.7)
    assert(secondRec.getAs[Seq[Double]](5) == List(0.2, 0.9))
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](6) ==
        List(Map("s3" -> 1.0), Map("s4" -> 1.0)))
    assert(secondRec.getAs[Seq[String]](7) == List("action3", "action4"))
    assert(secondRec.getAs[Long](8) == 11)
    assert(secondRec.getAs[Long](9) == 2)
    assert(secondRec.getAs[Seq[Long]](10) == List(1, 2))
    assert(secondRec.getAs[Seq[String]](11) == List("action2", "action3"))
    assert(
      secondRec.getAs[Seq[Seq[String]]](12) ==
        List(List("action3", "action4"), List("action4", "action1", "action2", "action3")))
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](13) ==
        List(Map("Widgets" -> 20.0), Map("Widgets" -> 30.0)))

    val thirdRec = df.where($"sequence_number_ordinal" === 3).head()
    assert(thirdRec.getAs[String](0) == "2018-01-01")
    assert(thirdRec.getAs[String](1) == "mdp1")
    assert(thirdRec.getAs[Map[String, Double]](2) == Map("s3" -> 1.0))
    assert(thirdRec.getAs[String](3) == "action3")
    assert(thirdRec.getAs[Double](4) == 0.6)
    assert(thirdRec.getAs[Seq[Double]](5) == List(0.9))
    assert(
      thirdRec.getAs[Seq[Map[String, Double]]](6) ==
        List(Map("s4" -> 1.0)))
    assert(thirdRec.getAs[Seq[String]](7) == List("action4"))
    assert(thirdRec.getAs[Long](8) == 12)
    assert(thirdRec.getAs[Long](9) == 3)
    assert(thirdRec.getAs[Seq[Long]](10) == List(1))
    assert(thirdRec.getAs[Seq[String]](11) == List("action3", "action4"))
    assert(
      thirdRec.getAs[Seq[Seq[String]]](12) ==
        List(List("action4", "action1", "action2", "action3")))
    assert(thirdRec.getAs[Seq[Map[String, Double]]](13) == List(Map("Widgets" -> 30.0)))
  }

  test("three-state-continuous-mdp-two-step-rl") {
    val action_discrete: Boolean = false
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = MultiStepTimelineConfiguration("2018-01-01",
                                                "2018-01-01",
                                                false,
                                                action_discrete,
                                                "some_rl_input",
                                                "some_rl_timeline",
                                                null,
                                                1,
                                                2)
    // destroy previous schema
    MultiStepTimeline.validateOrDestroyTrainingTable(sqlContext,
                                                     s"${config.outputTableName}",
                                                     action_discrete)

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          ("2018-01-01",
           "mdp1",
           1,
           1.0,
           Map("a1" -> 0.3, "a2" -> 0.5),
           0.8,
           Map("s1" -> 1.0),
           List(Map("a1" -> 0.3, "a2" -> 0.5), Map("a1" -> 0.6, "a2" -> 0.2)),
           Map("Widgets" -> 10.0)), // First state
          ("2018-01-01",
           "mdp1",
           11,
           0.2,
           Map("a1" -> 0.1, "a2" -> 0.9),
           0.7,
           Map("s2" -> 1.0),
           List(Map("a1" -> 0.1, "a2" -> 0.9), Map("a1" -> 0.8, "a2" -> 0.2)),
           Map("Widgets" -> 20.0)), // Second state
          ("2018-01-01",
           "mdp1",
           12,
           0.9,
           Map("a1" -> 0.8, "a2" -> 0.2),
           0.6,
           Map("s3" -> 1.0),
           List(Map("a1" -> 0.8, "a2" -> 0.2), Map("a1" -> 0.3, "a2" -> 0.3)),
           Map("Widgets" -> 30.0)) // Third state
        ))
      .toDF("ds",
            "mdp_id",
            "sequence_number",
            "reward",
            "action",
            "action_probability",
            "state_features",
            "possible_actions",
            "metrics")
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
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(sqlContext, s"${config.outputTableName}", action_discrete)
    )

    // Query the results
    println("----------------Input table----------------");
    val sqlInputCommand = s"""
         SELECT * FROM ${config.inputTableName}
         WHERE ds BETWEEN '${config.startDs}' AND '${config.endDs}'
    """.stripMargin
    sqlCtx.sql(sqlInputCommand).show(false);
    println("----------------Output table----------------");
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show(false)
    println("----------------Output table schema----------------");
    df.printSchema()
    assert(df.count() == 2)

    val firstRec = df.where($"sequence_number_ordinal" === 1).head()
    assert(firstRec.getAs[String](0) == "2018-01-01")
    assert(firstRec.getAs[String](1) == "mdp1")
    assert(firstRec.getAs[Map[String, Double]](2) == Map("s1" -> 1.0))
    assert(firstRec.getAs[Map[String, Double]](3) == Map("a1" -> 0.3, "a2" -> 0.5))
    assert(firstRec.getAs[Double](4) == 0.8)
    assert(firstRec.getAs[Seq[Double]](5) == List(1.0, 0.2))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](6) ==
        List(Map("s2" -> 1.0), Map("s3" -> 1.0)))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](7) ==
        List(Map("a1" -> 0.1, "a2" -> 0.9), Map("a1" -> 0.8, "a2" -> 0.2)))
    assert(firstRec.getAs[Long](8) == 1)
    assert(firstRec.getAs[Long](9) == 1)
    assert(firstRec.getAs[Seq[Long]](10) == List(10, 11))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](11) == List(Map("a1" -> 0.3, "a2" -> 0.5),
                                                           Map("a1" -> 0.6, "a2" -> 0.2)))
    assert(
      firstRec.getAs[Seq[Seq[Map[String, Double]]]](12) ==
        List(
          List(Map("a1" -> 0.1, "a2" -> 0.9), Map("a1" -> 0.8, "a2" -> 0.2)),
          List(Map("a1" -> 0.8, "a2" -> 0.2), Map("a1" -> 0.3, "a2" -> 0.3))
        ))
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](13) ==
        List(Map("Widgets" -> 10.0), Map("Widgets" -> 20.0)))

    val secondRec = df.where($"sequence_number_ordinal" === 2).head()
    assert(secondRec.getAs[String](0) == "2018-01-01")
    assert(secondRec.getAs[String](1) == "mdp1")
    assert(secondRec.getAs[Map[String, Double]](2) == Map("s2" -> 1.0))
    assert(secondRec.getAs[Map[String, Double]](3) == Map("a1" -> 0.1, "a2" -> 0.9))
    assert(secondRec.getAs[Double](4) == 0.7)
    assert(secondRec.getAs[Seq[Double]](5) == List(0.2))
    assert(secondRec.getAs[Seq[Map[String, Double]]](6) == List(Map("s3" -> 1.0)))
    assert(secondRec.getAs[Seq[Map[String, Double]]](7) == List(Map("a1" -> 0.8, "a2" -> 0.2)))
    assert(secondRec.getAs[Long](8) == 11)
    assert(secondRec.getAs[Long](9) == 2)
    assert(secondRec.getAs[Seq[Long]](10) == List(1))
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](11) == List(Map("a1" -> 0.1, "a2" -> 0.9),
                                                            Map("a1" -> 0.8, "a2" -> 0.2)))
    assert(
      secondRec.getAs[Seq[Seq[Map[String, Double]]]](12) ==
        List(List(Map("a1" -> 0.8, "a2" -> 0.2), Map("a1" -> 0.3, "a2" -> 0.3))))
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](13) ==
        List(Map("Widgets" -> 20.0)))
  }

  test("two-state-mdp") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration("2018-01-01",
                                       "2018-01-01",
                                       false,
                                       action_discrete,
                                       "some_rl_input",
                                       "some_rl_timeline",
                                       null,
                                       1)

    // destroy previous schema
    Timeline.validateOrDestroyTrainingTable(sqlContext,
                                            s"${config.outputTableName}",
                                            action_discrete)

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
           List("action1", "action2"),
           Map("Widgets" -> 10.0)), // First state
          ("2018-01-01",
           "mdp1",
           11,
           0.2,
           "action2",
           0.7,
           Map("s2" -> 1.0),
           List(),
           Map("Widgets" -> 20.0)) // Second state
        ))
      .toDF("ds",
            "mdp_id",
            "sequence_number",
            "reward",
            "action",
            "action_probability",
            "state_features",
            "possible_actions",
            "metrics")
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
    assert(
      Helper.outputTableIsValid(sqlContext, s"${config.outputTableName}", action_discrete)
    )

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
    assert(firstRow.getAs[Map[String, Double]](13) == Map("Widgets" -> 10.0))
  }
}
