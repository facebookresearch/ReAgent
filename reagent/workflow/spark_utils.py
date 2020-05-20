#!/usr/bin/env python3

import logging
import os
import pprint
from os.path import abspath, dirname, join
from typing import Dict, Optional

import reagent

# pyre-fixme[21]: Could not find `pyspark`.
# pyre-fixme[21]: Could not find `pyspark`.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


logger = logging.getLogger(__name__)

# This is where Scala preprocessing (i.e TimelineOperator) is located
SPARK_JAR_FROM_ROOT_DIR = "preprocessing/target/rl-preprocessing-1.1.jar"

"""
SPARK_JAR is abspath to the above jar file.

Assume file structure
ReAgent/
    preprocessing/...
    reagent/...
"""
SPARK_JAR = join(dirname(reagent.__file__), os.pardir, SPARK_JAR_FROM_ROOT_DIR)


DEFAULT_SPARK_CONFIG = {
    "spark.app.name": "ReAgent",
    "spark.sql.session.timeZone": "UTC",
    # use local host
    "spark.driver.host": "127.0.0.1",
    # use as many worker threads as possible on machine
    "spark.master": "local[*]",
    # default local warehouse for Hive
    "spark.sql.warehouse.dir": abspath("spark-warehouse"),
    # Set shuffle partitions to a low number, e.g. <= cores * 2 to speed
    # things up, otherwise the tests will use the default 200 partitions
    # and it will take a lot more time to complete
    "spark.sql.shuffle.partitions": "12",
    "spark.sql.execution.arrow.enabled": "true",
    # For accessing timeline operator
    "spark.driver.extraClassPath": SPARK_JAR,
    # Same effect as builder.enableHiveSupport() [useful for test framework]
    "spark.sql.catalogImplementation": "hive",
}


def get_spark_session(config: Optional[Dict[str, str]] = DEFAULT_SPARK_CONFIG):
    logger.info(f"Building with config: \n{pprint.pformat(config)}")
    spark = SparkSession.builder.enableHiveSupport()
    if config is not None:
        for k, v in config.items():
            spark = spark.config(k, v)
    spark = spark.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def call_spark_class(spark, class_name: str, args: str):
    spark_class = getattr(spark._jvm.com.facebook.spark.rl, class_name, None)
    assert spark_class is not None, f"Could not find {class_name}."
    spark_class.main(args)


def get_table_url(table_name: str) -> str:
    spark = get_spark_session()
    url = (
        spark.sql(f"DESCRIBE FORMATTED {table_name}")
        .filter((col("col_name") == "Location"))
        .select("data_type")
        .toPandas()
        .astype(str)["data_type"]
        .values[0]
    )
    # unfortunately url is file:/... or hdfs:/... not file:///...
    # so we need to insert '//'
    assert url.count(":") == 1, f"{url} has more than one :"
    schema, path = url.split(":")
    return f"{schema}://{path}"
