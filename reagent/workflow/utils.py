#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import Dict, List, Optional

# pyre-fixme[21]: Could not find module `pytorch_lightning`.
import pytorch_lightning as pl
import torch

# pyre-fixme[21]: Could not find `petastorm`.
from petastorm import make_batch_reader

# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
from petastorm.pytorch import DataLoader, decimal_friendly_collate

# pyre-fixme[21]: Could not find module `reagent.core.oss_tensorboard_logger`.
from reagent.core.oss_tensorboard_logger import OssTensorboardLogger

# pyre-fixme[21]: Could not find module `reagent.data.spark_utils`.
from reagent.data.spark_utils import get_spark_session

# pyre-fixme[21]: Could not find module `reagent.preprocessing.batch_preprocessor`.
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor

# pyre-fixme[21]: Could not find module `reagent.training`.
from reagent.training import StoppingEpochCallback

# pyre-fixme[21]: Could not find module `reagent.training.reagent_lightning_module`.
from reagent.training.reagent_lightning_module import has_test_step_override

from .types import Dataset, ReaderOptions, ResourceOptions


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    # pyre-fixme[16]: Module `reagent` has no attribute `data`.
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


# pyre-fixme[11]: Annotation `BatchPreprocessor` is not defined as a type.
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
    """get petastorm loader for dataset (with preprocessor)"""
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
# pyre-fixme[11]: Annotation `LightningDataModule` is not defined as a type.
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


def get_rank() -> int:
    """
    Returns the torch.distributed rank of the process. 0 represents
    the main process and is the default if torch.distributed isn't set up
    """
    return (
        # pyre-fixme[16]: Module `torch` has no attribute `distributed`.
        torch.distributed.get_rank()
        # pyre-fixme[16]: Module `torch` has no attribute `distributed`.
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


def train_eval_lightning(
    train_dataset,
    eval_dataset,
    test_dataset,
    trainer_module,
    data_module,
    num_epochs,
    logger_name: str,
    batch_preprocessor=None,
    reader_options: Optional[ReaderOptions] = None,
    checkpoint_path: Optional[str] = None,
    resource_options: Optional[ResourceOptions] = None,
    # pyre-fixme[11]: Annotation `Trainer` is not defined as a type.
) -> pl.Trainer:
    resource_options = resource_options or ResourceOptions()
    use_gpu = resource_options.use_gpu
    reader_options = reader_options or ReaderOptions()
    datamodule = data_module or PetastormLightningDataModule(
        train_dataset, eval_dataset, batch_preprocessor, reader_options
    )
    trainer = pl.Trainer(
        # pyre-fixme[16]: Module `reagent` has no attribute `core`.
        logger=OssTensorboardLogger(save_dir="pl_log_tensorboard", name=logger_name),
        max_epochs=num_epochs * 1000,
        gpus=int(use_gpu),
        reload_dataloaders_every_n_epochs=1,
        resume_from_checkpoint=checkpoint_path,
        # pyre-fixme[16]: Module `reagent` has no attribute `training`.
        callbacks=[StoppingEpochCallback(num_epochs)],
    )
    trainer.fit(trainer_module, datamodule=datamodule)
    # pyre-fixme[16]: Module `reagent` has no attribute `training`.
    if has_test_step_override(trainer_module):
        trainer.test(ckpt_path=None, datamodule=datamodule)
    else:
        logger.warning(
            f"Module {type(trainer_module).__name__} doesn't implement test_step(). Skipping testing"
        )
    if checkpoint_path is not None:
        # Overwrite the warmstart path with the new model
        trainer_module.set_clean_stop(True)
        trainer.save_checkpoint(checkpoint_path)
    return trainer
