package com.facebook.spark.rl

object Udfs {
  def unionListOfMaps[A, B](input: Seq[Map[A, B]]): Map[A, B] =
    input.flatten.toMap
}
