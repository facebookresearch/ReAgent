// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.common.testutil

import org.slf4j.{Logger, LoggerFactory}

trait TestLogger {
  lazy val log: Logger = LoggerFactory.getLogger(this.getClass.getName)
}
