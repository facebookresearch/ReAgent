#!/usr/bin/env python3

import logging
import time
from typing import List

from ml.rl.evaluation.cpe import CpeDetails
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.training._dqn_trainer import _DQNTrainer
from ml.rl.training.sac_trainer import SACTrainer
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.types import TrainingBatch


logger = logging.getLogger(__name__)


class PageHandler:
    def handle(self, tdp: TrainingDataPage) -> None:
        raise NotImplementedError()

    def finish(self) -> None:
        raise NotImplementedError()

    def set_epoch(self, epoch) -> None:
        raise NotImplementedError()


class TrainingPageHandler(PageHandler):
    def __init__(self, trainer):
        self.accumulated_tdp = None
        self.trainer = trainer

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def handle(self, tdp: TrainingDataPage) -> None:
        SummaryWriterContext.increase_global_step()
        self.trainer.train(tdp)

    def finish(self) -> None:
        self.trainer.loss_reporter.flush()


class EvaluationPageHandler(PageHandler):
    def __init__(self, trainer, evaluator, reporter):
        self.trainer = trainer
        self.evaluator = evaluator
        self.evaluation_data: EvaluationDataPage = None
        self.reporter = reporter
        self.results: List[CpeDetails] = []

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def handle(self, tdp: TrainingDataPage) -> None:
        if not self.trainer.calc_cpe_in_training:
            return
        if isinstance(tdp, TrainingDataPage):
            edp = EvaluationDataPage.create_from_tdp(tdp, self.trainer)
        elif isinstance(tdp, TrainingBatch):
            if isinstance(self.trainer, (_DQNTrainer, SACTrainer)):
                # TODO: Implement CPE for modular DQNTrainer & continuous algos
                edp = None
            else:
                edp = EvaluationDataPage.create_from_training_batch(tdp, self.trainer)
        if self.evaluation_data is None:
            self.evaluation_data = edp
        else:
            self.evaluation_data = self.evaluation_data.append(edp)

    def finish(self) -> None:
        if self.evaluation_data is None:
            return
        # Making sure the data is sorted for CPE
        self.evaluation_data = self.evaluation_data.sort()
        self.evaluation_data = self.evaluation_data.compute_values(self.trainer.gamma)
        self.evaluation_data.validate()
        start_time = time.time()
        evaluation_details = self.evaluator.evaluate_post_training(self.evaluation_data)
        self.reporter.report(evaluation_details)
        self.results.append(evaluation_details)
        logger.info("CPE evaluation took {} seconds.".format(time.time() - start_time))
        self.evaluation_data = None

    def get_last_cpe_results(self):
        if len(self.results) == 0:
            return CpeDetails()
        return self.results[-1]


def feed_pages(
    data_streamer,
    dataset_num_rows,
    epoch,
    minibatch_size,
    use_gpu,
    page_handler,
    feature_extractor=None,
    batch_preprocessor=None,
):
    num_rows_processed = 0
    num_rows_to_process_for_progress_tick = max(1, dataset_num_rows // 100)
    last_percent_reported = -1

    for batch in data_streamer:
        num_rows_processed += minibatch_size
        if (
            num_rows_processed // num_rows_to_process_for_progress_tick
        ) != last_percent_reported:
            last_percent_reported = (
                num_rows_processed // num_rows_to_process_for_progress_tick
            )
            logger.info(
                "Feeding page. Epoch: {}, Epoch Progress: {} of {} ({}%)".format(
                    epoch,
                    num_rows_processed,
                    dataset_num_rows,
                    (100 * num_rows_processed) // dataset_num_rows,
                )
            )

        # TODO: This preprocessing should go into background as well
        if batch_preprocessor:
            batch = batch_preprocessor(batch)

        page_handler.handle(batch)

    page_handler.finish()
