// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

object Udfs {
  def unionListOfMaps[A, B](input: Seq[Map[A, B]]): Map[A, B] =
    input.flatten.toMap
}
