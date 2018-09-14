package com.facebook.spark.rl

object Udfs {
  def unionListOfMaps[A, B](input: Seq[Map[A, B]]): Map[A, B] =
    input.flatten.toMap

  def getEpisodeValue[gamma](gamma: Double, m: Map[Long, Double]): Double = {
    val rewardMap = m.map { case (k, v) => (k, v * scala.math.pow(gamma, k.toInt)) }
    val sum: Double = 0
    rewardMap.foldLeft(sum)(_ + _._2)
  }
}
