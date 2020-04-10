#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.workflow.types import Dataset, TableSpec, ReaderOptions
from typing import Optional
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.workflow_utils.page_handler import PageHandler

from petastorm import make_reader, make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.workflow.spark_utils import get_spark_session
import logging

logger = logging.getLogger(__name__)



def get_table_row_count(parquet_url: str):
    with get_spark_session() as sc:
        return sc.read.parquet(str).count()

def collate_and_preprocess(batch_preprocessor: BatchPreprocessor):
    def collate_fn(batch):
        return batch_preprocessor(decimal_friendly_collate(batch))
    return collate_fn


# TODO: trainer only is used for minibatch?
def train_and_evaluate_generic(
    train_dataset: TableSpec,
    eval_dataset: Optional[TableSpec],
    trainer: RLTrainer,
    num_epochs: int,
    use_gpu: bool,
    batch_preprocessor: BatchPreprocessor,
    train_page_handler: PageHandler,
    eval_page_handler: PageHandler,
    reader_options: Optional[ReaderOptions] = None,
):
    if not reader_options:
        reader_options = ReaderOptions()

    train_dataset_num_rows = get_table_row_count(train_dataset.dataset.parquet_url)
    eval_dataset_num_rows = None
    # if eval_dataset is not None:
    # eval_dataset_num_rows = get_table_row_count(eval_dataset.parquet_url)
    logger.info(
        f"train_data_num: {train_dataset_num_rows}, eval_data_num: {eval_dataset_num_rows}"
    )

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch} start feeding training data")
        data_reader = make_batch_reader(
            train_dataset.dataset.parquet_url, num_epochs=1, reader_pool_type="thread"
        )
        # How to handle preprocessing?
        with DataLoader(data_reader, batch_size=trainer.minibatch_size, collate_fn=collate_and_preprocess(batch_preprocessor)) as data_loader:
            feed_pages(
                data_loader,
                train_dataset_num_rows,
                epoch,
                trainer.minibatch_size,
                use_gpu,
                train_page_handler,
            )

        # TODO: eval dataset
        # if not eval_dataset:
        #     continue

        # logger.info(f"Epoch {epoch} start feeding evaluation data")
        # data_loader = construct_data_loader(
        #     batch_preprocessor=batch_preprocessor,
        #     minibatch_size=trainer.minibatch_size,
        #     use_gpu=use_gpu,
        #     reader_options=reader_options,
        #     table_name=eval_dataset.table_name,
        # )
        # feed_pages(
        #     data_loader,
        #     eval_dataset_num_rows,
        #     epoch,
        #     trainer.minibatch_size,
        #     use_gpu,
        #     eval_page_handler,
        # )
