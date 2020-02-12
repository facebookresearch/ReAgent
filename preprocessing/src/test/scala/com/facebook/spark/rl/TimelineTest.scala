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
    val config = MultiStepTimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_1",
      "some_rl_timeline_1",
      null,
      1,
      1
    )
    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            "action1",
            0.8,
            Map(1L -> 1.0),
            List("action1", "action2"),
            Map("Widgets" -> 10.0)
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            "action2",
            0.7,
            Map(2L -> 1.0),
            List(),
            Map("Widgets" -> 20.0)
          ) // Second state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "string",
        Map("reward" -> "array<double>", "metrics" -> "array<map<string,double>>"),
        Map("possible_actions" -> "array<string>"),
        true
      )
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
    assert(firstRow.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRow.getAs[String](3) == "action1")
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Seq[Double]](5) === List(1.0))
    assert(firstRow.getAs[Seq[Map[Long, Double]]](6) == List(Map(2L -> 1.0)))
    assert(firstRow.getAs[Seq[String]](7) == List("action2"))
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Seq[Long]](10) == List(10))
    assert(firstRow.getAs[Long](11) == 0)
    assert(firstRow.getAs[Seq[String]](12) == List("action1", "action2"))
    // special case: possible_next_actions is a list (not list of list) when
    // possible_next_actions in all next n steps (here n = 1) are empty lists
    assert(firstRow.getAs[Seq[Seq[String]]](13) == List())
    assert(firstRow.getAs[Seq[Map[String, Double]]](14) == List(Map("Widgets" -> 10.0)))
  }

  test("four-state-discrete-mdp-three-step-rl") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = MultiStepTimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_2",
      "some_rl_timeline_2",
      null,
      1,
      3
    )
    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            "action1",
            0.8,
            Map(1L -> 1.0),
            List("action1", "action2"),
            Map("Widgets" -> 10.0)
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            "action2",
            0.7,
            Map(2L -> 1.0),
            List("action2", "action3"),
            Map("Widgets" -> 20.0)
          ), // Second state
          (
            "2018-01-01",
            "mdp1",
            12,
            0.9,
            "action3",
            0.6,
            Map(3L -> 1.0),
            List("action3", "action4"),
            Map("Widgets" -> 30.0)
          ), // Third state
          (
            "2018-01-01",
            "mdp1",
            13,
            0.4,
            "action4",
            0.5,
            Map(4L -> 1.0),
            List("action4", "action1", "action2", "action3"),
            Map("Widgets" -> 40.0)
          ) // Fourth state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "string",
        Map("reward" -> "array<double>", "metrics" -> "array<map<string,double>>"),
        Map("possible_actions" -> "array<string>"),
        true
      )
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
    assert(firstRec.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRec.getAs[String](3) == "action1")
    assert(firstRec.getAs[Double](4) == 0.8)
    assert(firstRec.getAs[Seq[Double]](5) == List(1.0, 0.2, 0.9))
    assert(
      firstRec.getAs[Seq[Map[Long, Double]]](6) ==
        List(Map(2L -> 1.0), Map(3L -> 1.0), Map(4L -> 1.0))
    )
    assert(firstRec.getAs[Seq[String]](7) == List("action2", "action3", "action4"))
    assert(firstRec.getAs[Long](8) == 1)
    assert(firstRec.getAs[Long](9) == 1)
    assert(firstRec.getAs[Seq[Long]](10) == List(10, 11, 12))
    assert(firstRec.getAs[Long](11) == 0)
    assert(firstRec.getAs[Seq[String]](12) == List("action1", "action2"))
    assert(
      firstRec.getAs[Seq[Seq[String]]](13) ==
        List(
          List("action2", "action3"),
          List("action3", "action4"),
          List("action4", "action1", "action2", "action3")
        )
    )
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](14) ==
        List(Map("Widgets" -> 10.0), Map("Widgets" -> 20.0), Map("Widgets" -> 30.0))
    )

    val secondRec = df.where($"sequence_number_ordinal" === 2).head()
    assert(secondRec.getAs[String](0) == "2018-01-01")
    assert(secondRec.getAs[String](1) == "mdp1")
    assert(secondRec.getAs[Map[Long, Double]](2) == Map(2L -> 1.0))
    assert(secondRec.getAs[String](3) == "action2")
    assert(secondRec.getAs[Double](4) == 0.7)
    assert(secondRec.getAs[Seq[Double]](5) == List(0.2, 0.9))
    assert(
      secondRec.getAs[Seq[Map[Long, Double]]](6) ==
        List(Map(3L -> 1.0), Map(4L -> 1.0))
    )
    assert(secondRec.getAs[Seq[String]](7) == List("action3", "action4"))
    assert(secondRec.getAs[Long](8) == 11)
    assert(secondRec.getAs[Long](9) == 2)
    assert(secondRec.getAs[Seq[Long]](10) == List(1, 2))
    assert(secondRec.getAs[Long](11) == 10)
    assert(secondRec.getAs[Seq[String]](12) == List("action2", "action3"))
    assert(
      secondRec.getAs[Seq[Seq[String]]](13) ==
        List(List("action3", "action4"), List("action4", "action1", "action2", "action3"))
    )
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](14) ==
        List(Map("Widgets" -> 20.0), Map("Widgets" -> 30.0))
    )

    val thirdRec = df.where($"sequence_number_ordinal" === 3).head()
    assert(thirdRec.getAs[String](0) == "2018-01-01")
    assert(thirdRec.getAs[String](1) == "mdp1")
    assert(thirdRec.getAs[Map[Long, Double]](2) == Map(3L -> 1.0))
    assert(thirdRec.getAs[String](3) == "action3")
    assert(thirdRec.getAs[Double](4) == 0.6)
    assert(thirdRec.getAs[Seq[Double]](5) == List(0.9))
    assert(
      thirdRec.getAs[Seq[Map[Long, Double]]](6) ==
        List(Map(4L -> 1.0))
    )
    assert(thirdRec.getAs[Seq[String]](7) == List("action4"))
    assert(thirdRec.getAs[Long](8) == 12)
    assert(thirdRec.getAs[Long](9) == 3)
    assert(thirdRec.getAs[Seq[Long]](10) == List(1))
    assert(thirdRec.getAs[Long](11) == 11)
    assert(thirdRec.getAs[Seq[String]](12) == List("action3", "action4"))
    assert(
      thirdRec.getAs[Seq[Seq[String]]](13) ==
        List(List("action4", "action1", "action2", "action3"))
    )
    assert(thirdRec.getAs[Seq[Map[String, Double]]](14) == List(Map("Widgets" -> 30.0)))
  }

  test("three-state-continuous-mdp-two-step-rl") {
    val action_discrete: Boolean = false
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = MultiStepTimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_3",
      "some_rl_timeline_3",
      null,
      1,
      2
    )
    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            Map(1001L -> 0.3, 1002L -> 0.5),
            0.8,
            Map(1L -> 1.0),
            List(Map(1001L -> 0.3, 1002L -> 0.5), Map(1001L -> 0.6, 1002L -> 0.2)),
            Map("Widgets" -> 10.0)
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            Map(1001L -> 0.1, 1002L -> 0.9),
            0.7,
            Map(2L -> 1.0),
            List(Map(1001L -> 0.1, 1002L -> 0.9), Map(1001L -> 0.8, 1002L -> 0.2)),
            Map("Widgets" -> 20.0)
          ), // Second state
          (
            "2018-01-01",
            "mdp1",
            12,
            0.9,
            Map(1001L -> 0.8, 1002L -> 0.2),
            0.6,
            Map(3L -> 1.0),
            List(Map(1001L -> 0.8, 1002L -> 0.2), Map(1001L -> 0.3, 1002L -> 0.3)),
            Map("Widgets" -> 30.0)
          ) // Third state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    MultiStepTimeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "map<bigint,double>",
        Map("reward" -> "array<double>", "metrics" -> "array<map<string,double>>"),
        Map(
          "possible_actions" -> "array<map<bigint,double>>"
        ),
        true
      )
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
    assert(firstRec.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRec.getAs[Map[Long, Double]](3) == Map(1001L -> 0.3, 1002L -> 0.5))
    assert(firstRec.getAs[Double](4) == 0.8)
    assert(firstRec.getAs[Seq[Double]](5) == List(1.0, 0.2))
    assert(
      firstRec.getAs[Seq[Map[Long, Double]]](6) ==
        List(Map(2L -> 1.0), Map(3L -> 1.0))
    )
    assert(
      firstRec.getAs[Seq[Map[Long, Double]]](7) ==
        List(Map(1001L -> 0.1, 1002L -> 0.9), Map(1001L -> 0.8, 1002L -> 0.2))
    )
    assert(firstRec.getAs[Long](8) == 1)
    assert(firstRec.getAs[Long](9) == 1)
    assert(firstRec.getAs[Seq[Long]](10) == List(10, 11))
    assert(firstRec.getAs[Long](11) == 0)
    assert(
      firstRec.getAs[Seq[Map[Long, Double]]](12) == List(
        Map(1001L -> 0.3, 1002L -> 0.5),
        Map(1001L -> 0.6, 1002L -> 0.2)
      )
    )
    assert(
      firstRec.getAs[Seq[Seq[Map[Long, Double]]]](13) ==
        List(
          List(Map(1001L -> 0.1, 1002L -> 0.9), Map(1001L -> 0.8, 1002L -> 0.2)),
          List(Map(1001L -> 0.8, 1002L -> 0.2), Map(1001L -> 0.3, 1002L -> 0.3))
        )
    )
    assert(
      firstRec.getAs[Seq[Map[String, Double]]](14) ==
        List(Map("Widgets" -> 10.0), Map("Widgets" -> 20.0))
    )

    val secondRec = df.where($"sequence_number_ordinal" === 2).head()
    assert(secondRec.getAs[String](0) == "2018-01-01")
    assert(secondRec.getAs[String](1) == "mdp1")
    assert(secondRec.getAs[Map[Long, Double]](2) == Map(2L -> 1.0))
    assert(secondRec.getAs[Map[Long, Double]](3) == Map(1001L -> 0.1, 1002L -> 0.9))
    assert(secondRec.getAs[Double](4) == 0.7)
    assert(secondRec.getAs[Seq[Double]](5) == List(0.2))
    assert(secondRec.getAs[Seq[Map[Long, Double]]](6) == List(Map(3L -> 1.0)))
    assert(secondRec.getAs[Seq[Map[Long, Double]]](7) == List(Map(1001L -> 0.8, 1002L -> 0.2)))
    assert(secondRec.getAs[Long](8) == 11)
    assert(secondRec.getAs[Long](9) == 2)
    assert(secondRec.getAs[Seq[Long]](10) == List(1))
    assert(secondRec.getAs[Long](11) == 10)
    assert(
      secondRec.getAs[Seq[Map[Long, Double]]](12) == List(
        Map(1001L -> 0.1, 1002L -> 0.9),
        Map(1001L -> 0.8, 1002L -> 0.2)
      )
    )
    assert(
      secondRec.getAs[Seq[Seq[Map[Long, Double]]]](13) ==
        List(List(Map(1001L -> 0.8, 1002L -> 0.2), Map(1001L -> 0.3, 1002L -> 0.3)))
    )
    assert(
      secondRec.getAs[Seq[Map[String, Double]]](14) ==
        List(Map("Widgets" -> 20.0))
    )
  }

  test("two-state-continuous-mdp") {
    val action_discrete: Boolean = false
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_4",
      "some_rl_timeline_4",
      null,
      1
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            Map(1001L -> 0.3, 1002L -> 0.5),
            0.8,
            Map(1L -> 1.0),
            List(Map(1001L -> 0.3, 1002L -> 0.5), Map(1001L -> 0.6, 1002L -> 0.2)),
            Map("Widgets" -> 10.0)
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            Map.empty[Long, Double],
            0.7,
            Map(2L -> 1.0),
            List(),
            Map("Widgets" -> 20.0)
          ) // Second state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "map<bigint,double>",
        Map("reward" -> "double", "metrics" -> "map<string,double>"),
        Map("possible_actions" -> "array<map<bigint,double>>")
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show(false)
    assert(df.count() == 1)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRow.getAs[Map[Long, Double]](3) == Map(1001L -> 0.3, 1002L -> 0.5))
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Double](5) === 1.0)
    assert(firstRow.getAs[Map[Long, Double]](6) == Map(2L -> 1.0))
    assert(firstRow.getAs[Map[Long, Double]](7).isEmpty)
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Long](10) == 10)
    assert(firstRow.getAs[Long](11) == 0)
    assert(
      firstRow.getAs[Seq[Map[Long, Double]]](12) == List(
        Map(1001L -> 0.3, 1002L -> 0.5),
        Map(1001L -> 0.6, 1002L -> 0.2)
      )
    )
    assert(firstRow.getAs[Seq[Map[Long, Double]]](13) == List())
    assert(firstRow.getAs[Map[String, Double]](14) == Map("Widgets" -> 10.0))
  }

  test("two-state-continuous-mdp-sparse-action") {
    val action_discrete: Boolean = false
    val extraFeatureColumns: List[String] =
      List("action_id_list_features", "action_id_score_list_features")
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_5",
      "some_rl_timeline_5",
      null,
      1,
      extraFeatureColumns = extraFeatureColumns
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            Map(1001L -> 0.3, 1002L -> 0.5),
            0.8,
            Map(1L -> 1.0),
            List(Map(1001L -> 0.3, 1002L -> 0.5), Map(1001L -> 0.6, 1002L -> 0.2)),
            Map("Widgets" -> 10.0),
            Map(35L -> List(156L, 157L), 36L -> List(138L)), // action_id_list_features
            Map(35L -> Map(156L -> 0.5, 157L -> 0.4), 36L -> Map(138L -> 0.3)) // action_id_score_list_features
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            Map.empty[Long, Double],
            0.7,
            Map(2L -> 1.0),
            List(),
            Map("Widgets" -> 20.0),
            Map(35L -> List(153L, 154L), 36L -> List(139L)), // action_id_list_features
            Map(35L -> Map(153L -> 0.1, 154L -> 0.2), 36L -> Map(139L -> 0.7)) // action_id_score_list_features
          ) // Second state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics",
        "action_id_list_features",
        "action_id_score_list_features"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    val rewardTypes = Map("reward" -> "double", "metrics" -> "map<string,double>")
    val extraFeatureDataTypes = Map(
      "possible_actions" -> "array<map<bigint,double>>",
      "action_id_list_features" -> "map<bigint,array<bigint>>",
      "action_id_score_list_features" -> "map<bigint,map<bigint,double>>"
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "map<bigint,double>",
        rewardTypes,
        extraFeatureDataTypes
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(
        s"""SELECT ${(Constants.TRAINING_DATA_COLUMN_NAMES ++ Constants.SPARSE_ACTION_COLUMN_NAMES)
          .mkString(",")} from ${config.outputTableName}"""
      )

    df.show(false)
    assert(df.count() == 1)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRow.getAs[Map[Long, Double]](3) == Map(1001L -> 0.3, 1002L -> 0.5))
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Double](5) === 1.0)
    assert(firstRow.getAs[Map[Long, Double]](6) == Map(2L -> 1.0))
    assert(firstRow.getAs[Map[Long, Double]](7).isEmpty)
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Long](10) == 10)
    assert(firstRow.getAs[Long](11) == 0)
    assert(
      firstRow.getAs[Seq[Map[Long, Double]]](12) == List(
        Map(1001L -> 0.3, 1002L -> 0.5),
        Map(1001L -> 0.6, 1002L -> 0.2)
      )
    )
    assert(firstRow.getAs[Seq[Map[Long, Double]]](13) == List())
    assert(firstRow.getAs[Map[String, Double]](14) == Map("Widgets" -> 10.0))
    assert(
      firstRow.getAs[Map[Long, Seq[Long]]](15)
        == Map(35L -> List(156L, 157L), 36L -> List(138L))
    )
    assert(
      firstRow.getAs[Map[Long, Map[Long, Double]]](16)
        == Map(35L -> Map(156L -> 0.5, 157L -> 0.4), 36L -> Map(138L -> 0.3))
    )
    assert(
      firstRow.getAs[Map[Long, Seq[Long]]](17)
        == Map(35L -> List(153L, 154L), 36L -> List(139L))
    )
    assert(
      firstRow.getAs[Map[Long, Map[Long, Double]]](18)
        == Map(35L -> Map(153L -> 0.1, 154L -> 0.2), 36L -> Map(139L -> 0.7))
    )
  }

  test("two-state-discrete-sparse-mdp") {
    val action_discrete: Boolean = true
    val extraFeatureColumns: List[String] =
      List("state_id_list_features", "state_id_score_list_features")
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_6",
      "some_rl_timeline_6",
      null,
      1,
      extraFeatureColumns = extraFeatureColumns
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            "action1",
            0.8,
            Map(1L -> 1.0),
            List("action1", "action2"),
            Map("Widgets" -> 10.0),
            Map(35L -> List(156L, 157L), 36L -> List(138L)), // state_id_list_features
            Map(35L -> Map(156L -> 0.5, 157L -> 0.4), 36L -> Map(138L -> 0.3)) // state_id_score_list_features
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            "action2",
            0.7,
            Map(2L -> 1.0),
            List(),
            Map("Widgets" -> 20.0),
            Map(35L -> List(153L, 154L), 36L -> List(139L)), // state_id_list_features
            Map(35L -> Map(153L -> 0.1, 154L -> 0.2), 36L -> Map(139L -> 0.7)) // state_id_score_list_features
          ) // Second state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics",
        "state_id_list_features",
        "state_id_score_list_features"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    val actionType = "string"
    val rewardTypes = Map("reward" -> "double", "metrics" -> "map<string,double>")
    val timelineJoinDataTypes = Map(
      "possible_actions" -> "array<string>",
      "state_id_list_features" -> "map<bigint,array<bigint>>",
      "state_id_score_list_features" -> "map<bigint,map<bigint,double>>"
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        actionType,
        rewardTypes,
        timelineJoinDataTypes
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(
        s"""SELECT ${(Constants.TRAINING_DATA_COLUMN_NAMES ++ Constants.SPARSE_DATA_COLUMN_NAMES)
          .mkString(",")} from ${config.outputTableName}"""
      )

    df.show(false)
    assert(df.count() == 1)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRow.getAs[String](3) == "action1")
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Double](5) === 1.0)
    assert(firstRow.getAs[Map[Long, Double]](6) == Map(2L -> 1.0))
    assert(firstRow.getAs[String](7) == "action2")
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Long](10) == 10)
    assert(firstRow.getAs[Long](11) == 0)
    assert(firstRow.getAs[Seq[String]](12) == List("action1", "action2"))
    assert(firstRow.getAs[Seq[String]](13) == List())
    assert(firstRow.getAs[Map[String, Double]](14) == Map("Widgets" -> 10.0))
    // sparse data columns:
    // state_id_list_features, state_id_score_list_features,
    // next_state_id_list_features, next_state_id_score_list_features
    assert(
      firstRow.getAs[Map[Long, Seq[Long]]](15)
        == Map(35L -> List(156L, 157L), 36L -> List(138L))
    )
    assert(
      firstRow.getAs[Map[Long, Map[Long, Double]]](16)
        == Map(35L -> Map(156L -> 0.5, 157L -> 0.4), 36L -> Map(138L -> 0.3))
    )
    assert(
      firstRow.getAs[Map[Long, Seq[Long]]](17)
        == Map(35L -> List(153L, 154L), 36L -> List(139L))
    )
    assert(
      firstRow.getAs[Map[Long, Map[Long, Double]]](18)
        == Map(35L -> Map(153L -> 0.1, 154L -> 0.2), 36L -> Map(139L -> 0.7))
    )
  }

  test("two-state-discrete-mdp") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_7",
      "some_rl_timeline_7",
      null,
      1
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01",
            "mdp1",
            1,
            1.0,
            "action1",
            0.8,
            Map(1L -> 1.0),
            List("action1", "action2"),
            Map("Widgets" -> 10.0)
          ), // First state
          (
            "2018-01-01",
            "mdp1",
            11,
            0.2,
            "action2",
            0.7,
            Map(2L -> 1.0),
            List(),
            Map("Widgets" -> 20.0)
          ) // Second state
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "string",
        Map("reward" -> "double", "metrics" -> "map<string,double>"),
        Map("possible_actions" -> "array<string>")
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show(false)
    assert(df.count() == 1)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Map[Long, Double]](2) == Map(1L -> 1.0))
    assert(firstRow.getAs[String](3) == "action1")
    assert(firstRow.getAs[Double](4) == 0.8)
    assert(firstRow.getAs[Double](5) === 1.0)
    assert(firstRow.getAs[Map[Long, Double]](6) == Map(2L -> 1.0))
    assert(firstRow.getAs[String](7) == "action2")
    assert(firstRow.getAs[Long](8) == 1)
    assert(firstRow.getAs[Long](9) == 1)
    assert(firstRow.getAs[Long](10) == 10)
    assert(firstRow.getAs[Long](11) == 0)
    assert(firstRow.getAs[Seq[String]](12) == List("action1", "action2"))
    assert(firstRow.getAs[Seq[String]](13) == List())
    assert(firstRow.getAs[Map[String, Double]](14) == Map("Widgets" -> 10.0))
  }

  test("filter-outliers") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    val percentileFunc = "percentile"

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      false,
      action_discrete,
      "some_rl_input_8",
      "some_rl_timeline_8",
      null,
      1,
      true,
      Some(0.95),
      percentileFunc
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        for (mdp_id <- 1 to 100; mdp_length = if (mdp_id <= 95) 2 else 100;
             seq_id <- 1 to mdp_length)
          yield
            if (mdp_id == 1 && seq_id == 1)
              (
                "2018-01-01",
                "mdp1",
                1,
                1.0,
                "action1",
                0.8,
                Map(1L -> 1.0),
                List("action1", "action2"),
                Map("Widgets" -> 10.0)
              ) // First state
            else
              (
                "2018-01-01",
                s"mdp${mdp_id}",
                seq_id,
                0.2,
                s"action${(seq_id + 1) % 2 + 1}",
                0.7,
                Map(2L -> 1.0),
                List("action1", "action2"),
                Map("Widgets" -> 20.0)
              ) // The other states
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "string",
        Map("reward" -> "double", "metrics" -> "map<string,double>"),
        Map("possible_actions" -> "array<string>")
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show()
    assert(df.count() == 95)
  }

  test("time-window-limit") {
    val action_discrete: Boolean = true
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    val percentileFunc = "percentile"

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      true,
      action_discrete,
      "some_rl_input_9",
      "some_rl_timeline_9",
      null,
      1,
      true,
      Some(0.95),
      percentileFunc,
      timeWindowLimit = Some(3L)
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        for (mdp_id <- 1 to 3; mdp_length = 10;
             seq_id <- 1 to mdp_length)
          yield (
            "2018-01-01",
            s"mdp${mdp_id}",
            seq_id,
            0.2,
            s"action${(seq_id + 1) % 2 + 1}",
            0.7,
            Map(2L -> 1.0),
            List("action1", "action2"),
            Map("Widgets" -> 20.0)
          )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "reward",
        "action",
        "action_probability",
        "state_features",
        "possible_actions",
        "metrics"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "string",
        Map("reward" -> "double", "metrics" -> "map<string,double>"),
        Map("possible_actions" -> "array<string>")
      )
    )

    // Query the results
    val df =
      sqlCtx.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")

    df.show()
    assert(df.count() == 12)
    df.collect.foreach(row => {
      assert(row.getAs[Long](11) <= 3)
      assert(row.getAs[Long](11) == row.getAs[Long](8) - 1)
    })

  }

  test("three-states-ranking-mdp") {
    val action_discrete: Boolean = false
    val sqlCtx = sqlContext
    import sqlCtx.implicits._
    val sparkContext = sqlCtx.sparkContext

    // Setup configuration
    val config = TimelineConfiguration(
      "2018-01-01",
      "2018-01-01",
      true,
      action_discrete,
      "some_rl_input_10",
      "some_rl_timeline_10",
      null,
      1,
      false,
      rewardColumns = List("slate_reward", "item_reward"),
      extraFeatureColumns = List("state_sequence_features")
    )

    // destroy previous schema
    Helper.destroyTrainingTable(
      sqlContext,
      s"${config.outputTableName}"
    )

    val state_sequence_features = Map(
      666L -> List(
        Map(1001L -> 0.3, 1002L -> 0.5),
        Map(1001L -> 0.9, 1002L -> 0.8),
        Map(1001L -> 0.1, 1002L -> 0.5),
        Map(1001L -> 0.6, 1002L -> 0.2)
      )
    )
    // Create fake input data
    val rl_input = sparkContext
      .parallelize(
        List(
          (
            "2018-01-01", // ds
            "mdp1", // mdp_id
            1, // sequence_number
            Map(123L -> 0.1, 234L -> 0.2), // slate_reward
            Map(123001L -> Map(0L -> 0.1, 1L -> 0.2), 234001L -> Map(0L -> 0.2, 3L -> 0.4)), // item_reward
            List(0L, 1L, 2L, 3L), // action
            0.8, // action_probability
            Map(555L -> 1.0), // state_features
            state_sequence_features
          ), // First slate
          (
            "2018-01-01", // ds
            "mdp1", // mdp_id
            2, // sequence_number
            Map(123L -> 0.3, 234L -> 0.2), // slate_reward
            Map(123001L -> Map(0L -> 0.1, 1L -> 0.2), 234001L -> Map(0L -> 0.2, 3L -> 0.4)), // item_reward
            List(3L, 1L, 2L, 0L), // action
            0.2, // action_probability
            Map(555L -> 2.0), // state_features
            state_sequence_features
          ), // second slate
          (
            "2018-01-01", // ds
            "mdp1", // mdp_id
            3, // sequence_number
            Map(123L -> 0.1, 234L -> 0.3), // slate_reward
            Map(123001L -> Map(0L -> 0.1, 1L -> 0.1), 234001L -> Map(0L -> 0.4, 3L -> 0.4)), // item_reward
            List(3L, 2L, 1L, 0L), // action
            0.5, // action_probability
            Map(555L -> 3.0), // state_features
            state_sequence_features
          ) // Third slate
        )
      )
      .toDF(
        "ds",
        "mdp_id",
        "sequence_number",
        "slate_reward",
        "item_reward",
        "action",
        "action_probability",
        "state_features",
        "state_sequence_features"
      )
    rl_input.createOrReplaceTempView(config.inputTableName)

    // Run the pipeline
    Timeline.run(sqlContext, config)

    // Ensure that the table is valid
    assert(
      Helper.outputTableIsValid(
        sqlContext,
        s"${config.outputTableName}",
        "array<bigint>",
        Map(
          "slate_reward" -> "map<bigint,double>",
          "item_reward" -> "map<bigint,map<bigint,double>>"
        ),
        Map(
          "state_sequence_features" -> "map<bigint,array<map<bigint,double>>>"
        )
      )
    )

    // Query the results
    val df =
      sqlCtx
        .sql(s"""SELECT ${Constants.RANKING_DATA_COLUMN_NAMES
          .mkString(",")} from ${config.outputTableName}""")
        .sort($"sequence_number".asc)

    df.show(false)
    assert(df.count() == 3)
    val firstRow = df.head
    assert(firstRow.getAs[String](0) == "2018-01-01")
    assert(firstRow.getAs[String](1) == "mdp1")
    assert(firstRow.getAs[Long](2) == 1L)
    assert(firstRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.1, 234L -> 0.2))
    assert(
      firstRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
        123001L -> Map(0L -> 0.1, 1L -> 0.2),
        234001L -> Map(0L -> 0.2, 3L -> 0.4)
      )
    )
    assert(firstRow.getAs[List[Long]](5) === List(0L, 1L, 2L, 3L))
    assert(firstRow.getAs[Double](6) == 0.8)
    assert(firstRow.getAs[Map[Long, Double]](7) == Map(555L -> 1.0))
    assert(firstRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
    assert(firstRow.getAs[List[Long]](9) === List(3L, 1L, 2L, 0L))
    assert(firstRow.getAs[Map[Long, Double]](10) == Map(555L -> 2.0))
    assert(firstRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == state_sequence_features)

    val secondRow = df.collect()(1)
    assert(secondRow.getAs[String](0) == "2018-01-01")
    assert(secondRow.getAs[String](1) == "mdp1")
    assert(secondRow.getAs[Long](2) == 2L)
    assert(secondRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.3, 234L -> 0.2))
    assert(
      secondRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
        123001L -> Map(0L -> 0.1, 1L -> 0.2),
        234001L -> Map(0L -> 0.2, 3L -> 0.4)
      )
    )
    assert(secondRow.getAs[List[Long]](5) === List(3L, 1L, 2L, 0L))
    assert(secondRow.getAs[Double](6) == 0.2)
    assert(secondRow.getAs[Map[Long, Double]](7) == Map(555L -> 2.0))
    assert(secondRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
    assert(secondRow.getAs[List[Long]](9) === List(3L, 2L, 1L, 0L))
    assert(secondRow.getAs[Map[Long, Double]](10) == Map(555L -> 3.0))
    assert(secondRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == state_sequence_features)

    val thirdRow = df.collect()(2)
    assert(thirdRow.getAs[String](0) == "2018-01-01")
    assert(thirdRow.getAs[String](1) == "mdp1")
    assert(thirdRow.getAs[Long](2) == 3L)
    assert(thirdRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.1, 234L -> 0.3))
    assert(
      thirdRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
        123001L -> Map(0L -> 0.1, 1L -> 0.1),
        234001L -> Map(0L -> 0.4, 3L -> 0.4)
      )
    )
    assert(thirdRow.getAs[List[Long]](5) === List(3L, 2L, 1L, 0L))
    assert(thirdRow.getAs[Double](6) == 0.5)
    assert(thirdRow.getAs[Map[Long, Double]](7) == Map(555L -> 3.0))
    assert(thirdRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
    assert(thirdRow.getAs[List[Long]](9) === List())
    assert(thirdRow.getAs[Map[Long, Double]](10) == Map())
    assert(thirdRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == Map())
  }

}
