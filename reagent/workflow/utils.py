#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.workflow.spark_utils import get_spark_session
from reagent.workflow.types import Dataset, ReaderOptions
from reagent.workflow_utils.page_handler import (
    EvaluationPageHandler,
    TrainingPageHandler,
    feed_pages,
)


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


# TODO: paralellize preprocessing by putting into transform of petastorm reader
def collate_and_preprocess(batch_preprocessor: BatchPreprocessor):
    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        return batch_preprocessor(batch)

    return collate_fn


def train_and_evaluate_generic(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    trainer: RLTrainer,
    num_epochs: int,
    use_gpu: bool,
    batch_preprocessor: BatchPreprocessor,
    train_page_handler: TrainingPageHandler,
    eval_page_handler: EvaluationPageHandler,
    reader_options: Optional[ReaderOptions] = None,
):
    reader_options = reader_options or ReaderOptions()

    train_dataset_num_rows = get_table_row_count(train_dataset.parquet_url)
    eval_dataset_num_rows = None
    if eval_dataset is not None:
        eval_dataset_num_rows = get_table_row_count(eval_dataset.parquet_url)

    logger.info(
        f"train_data_num: {train_dataset_num_rows}, "
        f"eval_data_num: {eval_dataset_num_rows}"
    )

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch} start feeding training data")
        data_reader = make_batch_reader(
            train_dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        with DataLoader(
            data_reader,
            batch_size=trainer.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor),
        ) as data_loader:
            feed_pages(
                data_loader,
                train_dataset_num_rows,
                epoch,
                trainer.minibatch_size,
                use_gpu,
                train_page_handler,
            )

        if not eval_dataset:
            continue

        logger.info(f"Epoch {epoch} start feeding evaluation data")
        eval_data_reader = make_batch_reader(
            eval_dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        with DataLoader(
            eval_data_reader,
            batch_size=trainer.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor),
        ) as eval_data_loader:
            feed_pages(
                eval_data_loader,
                eval_dataset_num_rows,
                epoch,
                trainer.minibatch_size,
                use_gpu,
                eval_page_handler,
            )
