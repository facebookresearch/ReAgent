#!/usr/bin/env python3

import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Dict, NamedTuple, Optional, Tuple

import torch
from reagent.core.types import (
    Dataset,
    ReaderOptions,
    RecurringPeriod,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.evaluation.evaluator import Evaluator
from reagent.parameters import NormalizationData
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.publishers.model_publisher import ModelPublisher
from reagent.tensorboardX import SummaryWriterContext, summary_writer_context
from reagent.training.trainer import Trainer
from reagent.validators.model_validator import ModelValidator
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow_utils.iterators import DataLoaderWrapper
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TrainEvalSampleRanges(NamedTuple):
    train_sample_range: Tuple[float, float]
    eval_sample_range: Tuple[float, float]


class BatchRunner:
    def __init__(
        self,
        use_gpu: bool,
        model_manager: ModelManager,
        data_fetcher: DataFetcher,
        reward_options: RewardOptions,
        normalization_data_map: Dict[str, NormalizationData],
        warmstart_path: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        self.model_manager = model_manager
        self.data_fetcher = data_fetcher
        self.normalization_data_map = normalization_data_map
        self.reward_options = reward_options
        self.warmstart_path = warmstart_path

    def get_workflow_id(self) -> int:
        raise NotImplementedError()

    def initialize_trainer(self) -> Trainer:
        # validate that we have all the required keys
        for normalization_key in self.model_manager.required_normalization_keys:
            normalization_data = self.normalization_data_map.get(
                normalization_key, None
            )
            assert normalization_data is not None, (
                f"NormalizationData for {normalization_key} "
                "is required but not provided."
            )
            # NOTE: Don't need this check in the future, for non-dense parameters
            assert normalization_data.dense_normalization_parameters is not None, (
                f"Dense normalization parameters for "
                f"{normalization_key} is not provided."
            )
        trainer = self.model_manager.build_trainer(
            self.use_gpu, self.normalization_data_map, self.reward_options
        )
        if self.warmstart_path is not None:
            trainer_state = torch.load(self.warmstart_path)
            trainer.load_state_dict(trainer_state)

        self.trainer = trainer
        return trainer

    def save_trainer(self, trainer: Trainer, output_path: str) -> None:
        """
        Save the trainer for warmstarting/checkpointing.
        """
        trainer_state = trainer.state_dict()
        torch.save(trainer_state, output_path)

    @staticmethod
    def get_sample_range(
        input_table_spec: TableSpec, calc_cpe_in_training: bool
    ) -> TrainEvalSampleRanges:
        table_sample = input_table_spec.table_sample
        eval_table_sample = input_table_spec.eval_table_sample

        if not calc_cpe_in_training:
            # use all data if table sample = None
            if table_sample is None:
                train_sample_range = (0.0, 100.0)
            else:
                train_sample_range = (0.0, table_sample)
            return TrainEvalSampleRanges(
                train_sample_range=train_sample_range,
                # eval samples will not be used
                eval_sample_range=(0.0, 0.0),
            )

        error_msg = (
            "calc_cpe_in_training is set to True. "
            f"Please specify table_sample(current={table_sample}) and "
            f"eval_table_sample(current={eval_table_sample}) such that "
            "eval_table_sample + table_sample <= 100. "
            "In order to reliably calculate CPE, eval_table_sample "
            "should not be too small."
        )
        assert table_sample is not None, error_msg
        assert eval_table_sample is not None, error_msg
        assert (eval_table_sample + table_sample) <= (100.0 + 1e-3), error_msg

        return TrainEvalSampleRanges(
            train_sample_range=(0.0, table_sample),
            eval_sample_range=(100.0 - eval_table_sample, 100.0),
        )

    def query(
        self,
        input_table_spec: TableSpec,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
    ) -> Tuple[Dataset, Dataset]:
        logger.info("Starting query")

        calc_cpe_in_training = self.model_manager.should_generate_eval_dataset
        sample_range_output = BatchRunner.get_sample_range(
            input_table_spec, calc_cpe_in_training
        )
        train_dataset = self.model_manager.query_data(
            data_fetcher=self.data_fetcher,
            input_table_spec=input_table_spec,
            sample_range=sample_range_output.train_sample_range,
            reward_options=self.reward_options,
        )
        eval_dataset = None
        if calc_cpe_in_training:
            eval_dataset = self.model_manager.query_data(
                data_fetcher=self.data_fetcher,
                input_table_spec=input_table_spec,
                sample_range=sample_range_output.eval_sample_range,
                reward_options=self.reward_options,
            )

        return (train_dataset, eval_dataset)

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        return self.model_manager.run_feature_identification(
            self.data_fetcher, input_table_spec
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        normalization_data_map: Dict[str, NormalizationData],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions] = None,
        warmstart_path: Optional[str] = None,
        validator: Optional[ModelValidator] = None,
        parent_workflow_id: Optional[int] = None,
        recurring_period: Optional[RecurringPeriod] = None,
    ) -> RLTrainingOutput:
        logger.info(f"{reader_options}")
        child_workflow_id = self.get_workflow_id()
        if parent_workflow_id is None:
            parent_workflow_id = child_workflow_id

        resource_options = resource_options or ResourceOptions()

        logger.info("Starting training")
        results = self.train_workflow(
            train_dataset,
            eval_dataset,
            num_epochs,
            parent_workflow_id=parent_workflow_id,
            child_workflow_id=child_workflow_id,
            reader_options=reader_options,
            resource_options=resource_options,
        )

        if validator is not None:
            results = self.run_validator(validator, results)

        return results

    def run_validator(
        self, model_validator: ModelValidator, training_output: RLTrainingOutput
    ) -> RLTrainingOutput:
        assert (
            training_output.validation_result is None
        ), f"validation_output was set to f{training_output.validation_output}"
        validation_result = model_validator.validate(training_output)
        return dataclasses.replace(training_output, validation_result=validation_result)

    def run_publisher(
        self,
        model_publisher: ModelPublisher,
        training_output: RLTrainingOutput,
        recurring_workflow_id: int,
        child_workflow_id: int,
        recurring_period: Optional[RecurringPeriod],
    ) -> RLTrainingOutput:
        assert (
            training_output.publishing_result is None
        ), f"publishing_output was set to f{training_output.publishing_output}"
        publishing_result = model_publisher.publish(
            self.model_manager,
            training_output,
            recurring_workflow_id,
            child_workflow_id,
            recurring_period,
        )
        return dataclasses.replace(training_output, publishing_result=publishing_result)

    def train_workflow(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_epochs: int,
        parent_workflow_id: int,
        child_workflow_id: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions] = None,
    ) -> RLTrainingOutput:
        writer = SummaryWriter()
        logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

        trainer = self.initialize_trainer()

        with summary_writer_context(writer):
            train_output: RLTrainingOutput = self._train(
                train_dataset, eval_dataset, num_epochs, reader_options, trainer
            )

        torchscript_output_path = f"model_{round(time.time())}.torchscript"
        serving_module = self.model_manager.build_serving_module(
            self.normalization_data_map, trainer
        )
        torch.jit.save(serving_module, torchscript_output_path)
        logger.info(f"Saved torchscript model to {torchscript_output_path}")
        return dataclasses.replace(
            train_output, local_output_path=torchscript_output_path
        )

    def _train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_epochs: int,
        reader_options: ReaderOptions,
        trainer: Trainer,
    ) -> RLTrainingOutput:
        reporter = self.model_manager.get_reporter()
        trainer.reporter = reporter

        evaluator = self.model_manager.get_evaluator(trainer, self.reward_options)
        if evaluator is not None:
            evaluator.reporter = reporter

        batch_preprocessor = self.model_manager.build_batch_preprocessor(
            reader_options,
            self.use_gpu,
            trainer.minibatch_size,
            self.normalization_data_map,
            self.reward_options,
        )
        return self.train_and_evaluate_generic(
            train_dataset,
            eval_dataset,
            trainer,
            num_epochs,
            self.use_gpu,
            batch_preprocessor,
            evaluator,
            reader_options,
        )

    def run_on_dataset_batches(
        self,
        run_on_batch_fn,
        dataset: Dataset,
        minibatch_size: int,
        batch_preprocessor: BatchPreprocessor,
        use_gpu: bool,
        reader_options: ReaderOptions,
        dataset_size: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        logger.info(f"{reader_options}")
        """ run_on_batch_fn is a function f that expects batches """
        if dataset_size is None:
            dataset_size = self.data_fetcher.get_table_row_count(dataset)
        assert dataset_size is not None
        assert dataset_size > 0, f"{dataset_size} is expected to be positive"

        @contextmanager
        def cleanup_dataloader_session(data_loader):
            try:
                yield data_loader
            finally:
                logger.info("Closing data loader")
                if hasattr(data_loader, "destroy_session"):
                    logger.info("Closing DistributedDataLoader")
                    data_loader.destroy_session()

        _dataloader = self.data_fetcher.get_dataloader(
            dataset=dataset,
            batch_size=minibatch_size,
            batch_preprocessor=batch_preprocessor,
            use_gpu=use_gpu,
            reader_options=reader_options,
        )
        with cleanup_dataloader_session(_dataloader) as dataloader:
            post_dataloader_preprocessor = self.data_fetcher.get_post_dataloader_preprocessor(
                reader_options=reader_options, use_gpu=use_gpu
            )
            dataloader_wrapper = DataLoaderWrapper(
                dataloader=dataloader,
                dataloader_size=dataset_size,
                post_dataloader_preprocessor=post_dataloader_preprocessor,
            )
            for batch in dataloader_wrapper:
                run_on_batch_fn(batch)
        return dataloader

    def train_and_evaluate_generic(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        trainer: Trainer,
        num_epochs: int,
        use_gpu: bool,
        batch_preprocessor: BatchPreprocessor,
        evaluator: Optional[Evaluator],
        reader_options: ReaderOptions,
        sort_eval_data: bool = True,
    ) -> RLTrainingOutput:
        logger.info(f"{reader_options}")
        assert num_epochs > 0, f"Epoch should be positive, got {num_epochs}"
        train_dataset_size = self.data_fetcher.get_table_row_count(train_dataset)
        if eval_dataset is not None and not sort_eval_data:
            eval_dataset_size = self.data_fetcher.get_table_row_count(eval_dataset)

        for epoch in range(num_epochs):
            SummaryWriterContext._reset_globals()
            logger.info(f"Starting training epoch {epoch}.")
            data_loader = self.run_on_dataset_batches(
                run_on_batch_fn=trainer.train,
                dataset=train_dataset,
                minibatch_size=trainer.minibatch_size,
                batch_preprocessor=batch_preprocessor,
                use_gpu=use_gpu,
                reader_options=reader_options,
                dataset_size=train_dataset_size,
            )
            if eval_dataset is not None and evaluator is not None:
                if sort_eval_data:
                    logger.info(
                        f"Starting evaluation epoch {epoch} by sorting and one shot"
                    )
                    eval_data = self.data_fetcher.gather_and_sort_eval_data(
                        trainer=trainer,
                        eval_dataset=eval_dataset,
                        batch_preprocessor=batch_preprocessor,
                        use_gpu=use_gpu,
                        reader_options=reader_options,
                    )
                    evaluator.evaluate_one_shot(eval_data)
                    evaluator.finish()
                else:
                    logger.info(
                        f"Starting evaluation epoch {epoch} by running on batches"
                    )
                    data_loader = self.run_on_dataset_batches(
                        run_on_batch_fn=evaluator.evaluate,
                        dataset=eval_dataset,
                        minibatch_size=trainer.minibatch_size,
                        batch_preprocessor=batch_preprocessor,
                        use_gpu=use_gpu,
                        reader_options=reader_options,
                        dataset_size=eval_dataset_size,
                    )
                    evaluator.finish()
            trainer.reporter.finish_epoch()
            report = trainer.reporter.publish()

        if hasattr(data_loader, "shutdown"):
            data_loader.shutdown()
        return report
