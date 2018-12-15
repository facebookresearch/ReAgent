// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
package com.facebook.spark.rl

object Udfs {
  def unionListOfMaps[A, B](input: Seq[Map[A, B]]): Map[A, B] =
    input.flatten.toMap

  def prepend[A](x: A, arr: Seq[A]): Seq[A] =
    x +: arr

  def sort_list_of_map[B](x: Seq[Map[Long, B]]): Seq[B] =
    x.sortBy(_.head._1).map(_.head._2)

  def drop_last[A](x: Seq[A]): Seq[A] =
    x.dropRight(1)

}
