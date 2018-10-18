// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.common.testutil

import com.google.common.collect.ImmutableList

import org.apache.log4j._
import org.apache.log4j.spi.LoggingEvent
import org.scalatest._

import scala.collection.JavaConversions._
import scala.util.Try

trait TestLogging extends BeforeAndAfterAll with BeforeAndAfterEach with TestLogger {
  this: Suite =>

  private val logLayout = new EnhancedPatternLayout("%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n")

  private var fullLogFile: Option[String] = None

  val SPARK_INTERNAL_NAMESPACES = ImmutableList
    .builder[String]
    .add("org.apache.spark")
    .add("org.apache.hadoop")
    .add("org.spark_project")
    .add("hive.ql")
    .add("DataNucleus")
    .build

  private def sanitize(s: String) = s.replaceAll("[\\/:*?\"<>|$ ]", "_")

  private def getTestSuiteLogFile = s"/tmp/${getClass.getName}.log"

  private def getTestCaseLogFile(testCase: String) =
    s"/tmp/${getClass.getName}_${sanitize(testCase)}.log"

  override def beforeAll(): Unit = {
    clearAllAppenders()
    addConsoleAppender
    addFileAppender(getTestSuiteLogFile)
    log.info(s"Full logs for test suite: ${getTestSuiteLogFile}")
    super.beforeAll
  }

  override def afterAll(): Unit = {
    super.afterAll
    removeAppender(getTestSuiteLogFile)
    removeAppender("console")
  }

  // We override withFixture because we want to setup the logging *AFTER* the IntelliJ test case console window is
  // created and directly before the test starts running.  If the logging is setup in BeforeEach then it logs into
  // the test suite console rather than the test case console.
  abstract override def withFixture(test: NoArgTest): Outcome = {
    val wrappedTest = new NoArgTest {
      val name = test.name
      def apply(): Outcome = {
        val testCaseLogFile = getTestCaseLogFile(test.name)
        Try {
          addFileAppender(testCaseLogFile)
          Thread.sleep(1)
          log.info(s"Full logs for test case: ${testCaseLogFile}")
        }
        try {
          test.apply
        } finally {
          Try {
            removeAppender(testCaseLogFile)
          }
        }
      }
      val configMap = test.configMap
      val scopes = test.scopes
      val text = test.text
      val tags = test.tags
    }

    super.withFixture(wrappedTest)
  }

  private def clearAllAppenders(): Unit =
    org.apache.log4j.Logger.getRootLogger.removeAllAppenders()

  def addFileAppender(logFile: String): Unit = {
    val fileAppender = new FileAppender
    fileAppender.setName(logFile)
    fileAppender.setFile(logFile)
    fileAppender.setLayout(logLayout)
    fileAppender.setThreshold(Level.ALL)
    fileAppender.setAppend(false)
    fileAppender.activateOptions
    fileAppender.setImmediateFlush(true)

    val rootLogger = org.apache.log4j.Logger.getRootLogger
    rootLogger.addAppender(fileAppender)
  }

  def addConsoleAppender(): Unit = {
    val conAppender = new ConsoleAppender
    conAppender.setName("console")
    conAppender.setTarget("System.err")
    conAppender.setLayout(logLayout)
    conAppender.setThreshold(Level.INFO)
    conAppender.activateOptions

    conAppender.addFilter(new spi.Filter {
      override def decide(event: LoggingEvent): Int = {
        val loggerName = event.getLoggerName
        if (loggerName != null && SPARK_INTERNAL_NAMESPACES.exists(loggerName.startsWith(_)))
          return -1

        return 1
      }
    })

    val rootLogger = org.apache.log4j.Logger.getRootLogger
    rootLogger.addAppender(conAppender)
  }

  def removeAppender(name: String): Unit = {
    val rootLogger = org.apache.log4j.Logger.getRootLogger
    rootLogger.removeAppender(name)
  }

  def ensureLoggerFilters(): Unit = {
    val rootLogger = org.apache.log4j.Logger.getRootLogger
    val loggerRepository = rootLogger.getLoggerRepository

    loggerRepository.getLogger("org.spark_project.jetty").setLevel(Level.WARN)
    loggerRepository
      .getLogger("org.spark_project.jetty.util.component.AbstractLifeCycle")
      .setLevel(Level.ERROR)
    loggerRepository
      .getLogger("org.apache.hadoop.hive.metastore.RetryingHMSHandler")
      .setLevel(Level.FATAL)
    loggerRepository
      .getLogger("org.apache.hadoop.hive.ql.exec.FunctionRegistry")
      .setLevel(Level.ERROR)
  }

  def showAppenders(): Unit = {
    val rootLogger = org.apache.log4j.Logger.getRootLogger
    rootLogger.getAllAppenders.foreach(appender => {
      val origAppender = appender.asInstanceOf[Appender]
      log.info(
        s"\33[31m Appender [${origAppender.getName}] [${origAppender.getClass.getName}] [${origAppender.getLayout}]")
    })
  }

  def showLoggers(): Unit = {
    val rootLogger = org.apache.log4j.Logger.getRootLogger
    rootLogger.getLoggerRepository.getCurrentLoggers
      .foreach(logger => {
        val log4jLogger = logger.asInstanceOf[org.apache.log4j.Logger]
        log.info(
          s"\33[35m Logger [${log4jLogger.getName}] [${log4jLogger.getParent.getName}] [${log4jLogger.getLevel}] [${log4jLogger.getEffectiveLevel}] [${log4jLogger.getAdditivity}]")
      })
  }
}
