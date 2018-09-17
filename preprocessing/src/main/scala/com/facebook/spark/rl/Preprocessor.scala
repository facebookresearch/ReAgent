package com.facebook.spark.rl

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

case class Configuration(timeline: TimelineConfiguration, query: QueryConfiguration)

case class QueryConfiguration(discountFactor: Double = 0.9,
                              tableSample: Double = 1.0,
                              maxQLearning: Boolean = true,
                              useNonOrdinalRewardTimeline: Boolean = false,
                              actions: Array[String] = Array())

object Preprocessor {
  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()
    sparkSession.sqlContext.udf.register("COMPUTE_EPISODE_VALUE", Udfs.getEpisodeValue[Double] _)

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
        possibleActionSchema
      ))

    val inputDf = sparkSession.read.schema(schema).json(timelineConfig.inputTableName)
    inputDf.createOrReplaceTempView(timelineConfig.inputTableName)

    Timeline.run(sparkSession.sqlContext, timelineConfig)
    val query = if (timelineConfig.actionDiscrete) {
      Query.getDiscreteQuery(queryConfig)
    } else {
      Query.getContinuousQuery(queryConfig)
    }

    // Query the results
    val outputDf = sparkSession.sql(
      query.concat(
        s" FROM ${timelineConfig.outputTableName} where rand() <= ${queryConfig.tableSample}"))

    outputDf.show()
    outputDf.write.json(timelineConfig.outputTableName)
    sparkSession.stop()
  }
}
