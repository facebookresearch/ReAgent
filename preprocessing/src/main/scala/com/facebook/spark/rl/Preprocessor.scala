package com.facebook.spark.rl

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object Preprocessor {
  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()

    val configJson = args(0)

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)
    val config = mapper.readValue(configJson, classOf[TimelineConfiguration])

    val schema = StructType(
      List(
        StructField("ds", StringType),
        StructField("mdp_id", StringType),
        StructField("sequence_number", IntegerType),
        StructField("action_probability", DoubleType),
        StructField("state_features", MapType(StringType, DoubleType, true)),
        StructField("action", StringType),
        StructField("reward", DoubleType),
        StructField("possible_actions", ArrayType(StringType))
      ))
    val inputDf = sparkSession.read.schema(schema).json(config.inputTableName)
    inputDf.createOrReplaceTempView(config.inputTableName)

    Timeline.run(sparkSession.sqlContext, config)

    // Query the results
    val outputDf =
      sparkSession.sql(s"""SELECT ${Constants.TRAINING_DATA_COLUMN_NAMES
        .mkString(",")} from ${config.outputTableName}""")
    outputDf.show()
    outputDf.write.json(config.outputTableName)
    sparkSession.stop()
  }
}
