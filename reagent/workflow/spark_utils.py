#!/usr/bin/env python3

import pprint
from os.path import abspath
from typing import Dict, Optional

from pyspark.sql import SparkSession


DEFAULT_SPARK_CONFIG = {
    "spark.master": "local[1]",
    "spark.app.name": "ReAgent",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.warehouse.dir": abspath("spark-warehouse"),
    "spark.sql.shuffle.partitions": "12",
}


def get_spark_session(config: Optional[Dict[str, str]] = DEFAULT_SPARK_CONFIG):
    print("Building with config: ")
    pprint.pprint(config)
    spark = SparkSession.builder.enableHiveSupport()
    if config is not None:
        for k, v in config.items():
            spark = spark.config(k, v)
    spark = spark.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
