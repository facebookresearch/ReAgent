#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import reagent.types as rlt

# pyre-fixme[21]: Could not find `petastorm`.
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.core.tracker import Observer
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import Evaluator
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.torch_utils import dict_to_tensor
from reagent.training import RLTrainer, SACTrainer, TD3Trainer
from reagent.workflow.spark_utils import get_spark_session
from reagent.workflow.types import Dataset, ReaderOptions
from reagent.workflow_utils.iterators import DataLoaderWrapper, EpochIterator


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


def collate_and_preprocess(batch_preprocessor: BatchPreprocessor, use_gpu: bool):
    """ Helper for Petastorm's DataLoader to preprocess.
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
    # NOTE: must be wrapped by DataLoaderWrapper to call __exit__() on end of epoch
    return DataLoader(
        data_reader,
        batch_size=batch_size,
        collate_fn=collate_and_preprocess(
            batch_preprocessor=batch_preprocessor, use_gpu=use_gpu
        ),
    )


def gather_eval_data(
    trainer: RLTrainer,
    eval_dataset: Dataset,
    batch_preprocessor: BatchPreprocessor,
    use_gpu: bool,
    reader_options: ReaderOptions,
) -> EvaluationDataPage:
    """ Sorts, computes logged values and validates the EvaluationDataPage """
    if isinstance(trainer, (SACTrainer, TD3Trainer)):
        raise NotImplementedError("TODO: Implement CPE for continuous algos")
    assert (
        trainer.calc_cpe_in_training
    ), "this function should only be called when this is true."

    # first read the eval_dataset as EvaluationDataPages
    device = "cuda" if use_gpu else "cpu"
    eval_data = None
    with make_batch_reader(
        eval_dataset.parquet_url,
        num_epochs=1,
        reader_pool_type=reader_options.petastorm_reader_pool_type,
    ) as reader:
        for batch in reader:
            assert rlt.isinstance_namedtuple(batch)
            tensor_batch = dict_to_tensor(batch._asdict(), device=device)
            # pyre-fixme[9]: tdp has type `PreprocessedTrainingBatch`; used as
            #  `TensorDataClass`.
            tdp: rlt.PreprocessedTrainingBatch = batch_preprocessor(tensor_batch)
            edp = EvaluationDataPage.create_from_training_batch(tdp, trainer)
            if eval_data is None:
                eval_data = edp
            else:
                eval_data = eval_data.append(edp)

    eval_data = eval_data.sort()
    eval_data = eval_data.compute_values(trainer.gamma)
    eval_data.validate()
    return eval_data


def train_and_evaluate_generic(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    trainer: RLTrainer,
    num_epochs: int,
    use_gpu: bool,
    batch_preprocessor: BatchPreprocessor,
    reporter: Observer,
    evaluator: Evaluator,
    reader_options: Optional[ReaderOptions] = None,
) -> None:
    reader_options = reader_options or ReaderOptions()
    epoch_iterator = EpochIterator(num_epochs=num_epochs)
    train_dataset_size = get_table_row_count(train_dataset.parquet_url)
    # pyre-fixme[16]: `EpochIterator` has no attribute `add_observer`.
    for epoch in epoch_iterator.add_observer(reporter):
        logger.info(f"Starting training epoch {epoch}.")
        dataloader = get_petastorm_dataloader(
            dataset=train_dataset,
            # pyre-fixme[6]: Expected `int` for 2nd param but got `Optional[int]`.
            batch_size=trainer.minibatch_size,
            batch_preprocessor=batch_preprocessor,
            use_gpu=use_gpu,
            reader_options=reader_options,
        )
        dataloader_wrapper = DataLoaderWrapper(
            dataloader=dataloader, dataloader_size=train_dataset_size
        )
        for batch in dataloader_wrapper:
            trainer.train(batch)

        if eval_dataset is not None:
            eval_data = gather_eval_data(
                trainer=trainer,
                eval_dataset=eval_dataset,
                batch_preprocessor=batch_preprocessor,
                use_gpu=use_gpu,
                reader_options=reader_options,
            )
            # evaluator passes cpe_details to reporter via notify_observers
            evaluator.evaluate_post_training(eval_data)
