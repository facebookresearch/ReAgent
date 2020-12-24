#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import pytorch_lightning as pl

# pyre-fixme[21]: Could not find `petastorm`.
from petastorm import make_batch_reader

# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.training import StoppingEpochCallback

from .spark_utils import get_spark_session
from .types import Dataset, ReaderOptions


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


def collate_and_preprocess(batch_preprocessor: BatchPreprocessor, use_gpu: bool):
    """Helper for Petastorm's DataLoader to preprocess.
    TODO(kaiwenw): parallelize preprocessing by using transform of Petastorm reader
    Should pin memory and preprocess in reader and convert to gpu in collate_fn.
    """

    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch

    return collate_fn


def get_petastorm_dataloader(
    dataset: Dataset,
    batch_size: int,
    batch_preprocessor: BatchPreprocessor,
    use_gpu: bool,
    reader_options: ReaderOptions,
):
    """ get petastorm loader for dataset (with preprocessor) """
    data_reader = make_batch_reader(
        dataset.parquet_url,
        num_epochs=1,
        reader_pool_type=reader_options.petastorm_reader_pool_type,
    )
    return DataLoader(
        data_reader,
        batch_size=batch_size,
        collate_fn=collate_and_preprocess(
            batch_preprocessor=batch_preprocessor, use_gpu=use_gpu
        ),
    )



# TODO: Move this to appropriate location
class PetastormLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, batch_preprocessor, reader_options):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_preprocessor = batch_preprocessor
        self.reader_options = reader_options

    def _closing_iter(self, dataloader):
        yield from dataloader
        dataloader.__exit__(None, None, None)

    def train_dataloader(self):
        dataloader = get_petastorm_dataloader(
            dataset=self.train_dataset,
            batch_size=self.reader_options.minibatch_size,
            batch_preprocessor=self.batch_preprocessor,
            use_gpu=False,
            reader_options=self.reader_options,
        )
        return self._closing_iter(dataloader)

    def test_dataloader(self):
        dataloader = get_petastorm_dataloader(
            dataset=self.eval_dataset,
            batch_size=self.reader_options.minibatch_size,
            batch_preprocessor=self.batch_preprocessor,
            use_gpu=False,
            reader_options=self.reader_options,
        )
        return self._closing_iter(dataloader)


def train_eval_lightning(
    train_dataset,
    eval_dataset,
    trainer_module,
    data_module,
    num_epochs,
    use_gpu,
    batch_preprocessor=None,
    reader_options: Optional[ReaderOptions] = None,
    checkpoint_path: Optional[str] = None,
) -> pl.Trainer:
    reader_options = reader_options or ReaderOptions()
    datamodule = data_module or PetastormLightningDataModule(
        train_dataset, eval_dataset, batch_preprocessor, reader_options
    )
    # pyre-fixme[16]: Module `pl` has no attribute `Trainer`.
    # pyre-fixme[16]: Module `pl` has no attribute `Trainer`.
    trainer = pl.Trainer(
        max_epochs=num_epochs * 1000,
        gpus=int(use_gpu),
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=checkpoint_path,
        callbacks=[StoppingEpochCallback(num_epochs)],
    )
    trainer.fit(trainer_module, datamodule=datamodule)
    trainer.test()
    return trainer
