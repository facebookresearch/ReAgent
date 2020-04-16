#!/usr/bin/env python3

import pyspark


LOCAL_MASTER = "local[1]"


def get_spark_session(master: str = LOCAL_MASTER):
    spark = (
        pyspark.sql.SparkSession.builder.master(master)
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark
