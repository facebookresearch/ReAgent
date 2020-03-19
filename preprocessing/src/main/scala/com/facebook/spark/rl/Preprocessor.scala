// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

import org.slf4j.LoggerFactory
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

case class Configuration(timeline: TimelineConfiguration, query: QueryConfiguration)

case class QueryConfiguration(tableSample: Double = 1.0, actions: Array[String] = Array())

object Preprocessor {

  private val log = LoggerFactory.getLogger(this.getClass.getName)

  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()

    val configJson = args(0)

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)
    val config = mapper.readValue(configJson, classOf[Configuration])
    val timelineConfig = config.timeline
    val queryConfig = config.query

    val actionSchema = if (timelineConfig.actionDiscrete) {
      StructField("action", StringType)
    } else {
      StructField("action", MapType(StringType, DoubleType, true))
    }
    val possibleActionSchema = if (timelineConfig.actionDiscrete) {
      StructField("possible_actions", ArrayType(StringType))
    } else {
      StructField("possible_actions", ArrayType(MapType(StringType, DoubleType, true)))
    }

    val schema = StructType(
      List(
        StructField("ds", StringType),
        StructField("mdp_id", StringType),
        StructField("sequence_number", IntegerType),
        StructField("action_probability", DoubleType),
        StructField("state_features", MapType(StringType, DoubleType, true)),
        actionSchema,
        StructField("reward", DoubleType),
        possibleActionSchema,
        StructField("metrics", MapType(StringType, DoubleType, true))
      )
    )

    var inputDf = sparkSession.read.schema(schema).json(timelineConfig.inputTableName)

    val mapStringDoubleToLongDouble = udf(
      (r: Map[String, Double]) => r.map({ case (key, value) => (key.toLong, value) })
    )

    inputDf = inputDf.withColumn(
      "state_features",
      mapStringDoubleToLongDouble(inputDf.col("state_features"))
    )
    if (!timelineConfig.actionDiscrete) {
      inputDf = inputDf.withColumn("action", mapStringDoubleToLongDouble(inputDf.col("action")))

      val mapArrayStringDoubleToArrayLongDouble = udf(
        (r: Array[Map[String, Double]]) =>
          r.map((m) => m.map({ case (key, value) => (key.toLong, value) }))
      )
      inputDf = inputDf.withColumn(
        "possible_actions",
        mapArrayStringDoubleToArrayLongDouble(inputDf.col("possible_actions"))
      )
    }

    inputDf.createOrReplaceTempView(timelineConfig.inputTableName)

    val tmpTableName = "tmp_table_for_preprocessor"
    Timeline.run(sparkSession.sqlContext, timelineConfig, Some(tmpTableName))
    val query = if (timelineConfig.actionDiscrete) {
      Query.getDiscreteQuery(queryConfig)
    } else {
      Query.getContinuousQuery(queryConfig)
    }

    val sqlCommand = query.concat(
      s" FROM ${tmpTableName} where ABS(HASH(mdp_id || 'train')) % 10000 <= CAST(${queryConfig.tableSample} * 100 AS INTEGER)"
    )
    log.info("Executing query: ")
    log.info(sqlCommand)
    val outputDf = sparkSession.sql(sqlCommand).repartition(timelineConfig.numOutputShards)
    outputDf.show()
    outputDf
      .write
      .mode(SaveMode.Overwrite)
      .json(timelineConfig.outputTableName)
    outputDf.write.saveAsTable(timelineConfig.outputTableName)

    val evalSqlCommand =
      query.concat(
        s" FROM ${tmpTableName} where ABS(HASH(mdp_id || 'eval')) % 10000 <= CAST(${queryConfig.tableSample / 10.0} * 100 AS INTEGER) ORDER BY mdp_id, sequence_number"
      )
    log.info("Executing query: ")
    log.info(evalSqlCommand)
    val evalOutputDf = sparkSession.sql(evalSqlCommand).repartition(timelineConfig.numOutputShards)
    evalOutputDf.show()
    evalOutputDf
      .write
      .mode(SaveMode.Overwrite)
      .json(timelineConfig.evalTableName)
    evalOutputDf.write.saveAsTable(timelineConfig.evalTableName)

    // this table was a temporary proxy for the above two commands.
    sparkSession.sql(s"DROP TABLE ${tmpTableName}")
    sparkSession.stop()
  }
}
