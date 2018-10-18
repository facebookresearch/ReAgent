// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.common.testutil

import java.io.File
import java.util.Date

import org.apache.hadoop.hive.conf.HiveConf
import org.apache.hadoop.hive.conf.HiveConf.ConfVars
import org.apache.hadoop.hive.metastore._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.scalatest.{BeforeAndAfterAll, FunSuiteLike, Suite}

import scala.collection.mutable
import scala.math.abs

trait PipelineTester extends FunSuiteLike with BeforeAndAfterAll with TestLogging { this: Suite =>

  @transient private var _sparkContext: SparkContext = _
  def sparkContext: SparkContext = _sparkContext
  def sc: SparkContext = _sparkContext

  @transient private var _sparkSession: SparkSession = _
  def sparkSession: SparkSession = _sparkSession

  @transient private var _sqlContext: SQLContext = _
  def sqlContext: SQLContext = _sqlContext

  val appID = new Date().toString + math.floor(math.random * 10E4).toLong.toString

  val sparkConf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("unit test")
    .set("spark.ui.enabled", "false")
    .set("spark.ui.showConsoleProgress", "false")
    .set("spark.app.id", appID)
    .set("spark.driver.host", "localhost")
    .set("spark.depa.enabled", "false")
    .set("spark.sql.catalogImplementation", "hive")

  override def beforeAll() {
    super.beforeAll

    val sparkSession = SparkSession.builder().config(sparkConf).enableHiveSupport().getOrCreate()
    val sparkContext = sparkSession.sparkContext

    _sqlContext = sparkSession.sqlContext
    _sparkContext = sparkSession.sparkContext
    _sparkSession = sparkSession

    _sqlContext.sql("set spark.sql.shuffle.partitions=2")

    _sparkContext.setLogLevel(org.apache.log4j.Level.ALL.toString)
  }

  override def afterAll() {
    try {
      _sparkSession.stop()
    } finally {
      super.afterAll()
    }
  }
}

object PipelineTester {}
