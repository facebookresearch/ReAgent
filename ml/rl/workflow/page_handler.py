#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from ml.rl.core.tracker import observable
from ml.rl.evaluation.cpe import CpeDetails
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.training.sac_trainer import SACTrainer
from ml.rl.training.td3_trainer import TD3Trainer
from ml.rl.types import (
    ExtraData,
    PreprocessedMemoryNetworkInput,
    PreprocessedTrainingBatch,
    RawTrainingBatch,
)


logger = logging.getLogger(__name__)


class PageHandler:
    def __init__(self, trainer_or_evaluator):
        self.trainer_or_evaluator = trainer_or_evaluator
        self.results: List[Dict] = []
        self.epoch = 0

    def refresh_results(self) -> None:
        self.results: List[Dict] = []

    def get_loss(self, loss_name="loss"):
        """ See usage in get_mean_loss """
        return [float(result[loss_name]) for result in self.results]

    def get_mean_loss(self, loss_name="loss", axis=None):
        """
        Get the average of a certain type of loss

        :param loss_name: possible loss names:
        For world model:
            'loss' (referring to total loss),
            'bce' (loss for predicting not_terminal),
            'gmm' (loss for next state prediction),
            'mse' (loss for predicting reward)
        For ranking model:
            'pg' (policy gradient loss)
            'baseline' (the baseline model's loss, usually for fitting V(s))
            'kendall_tau' (kendall_tau coefficient between advantage and log_probs,
             used in evaluation page handlers)
            'kendaull_tau_p_value' (the p-value for kendall_tau test, used in
             evaluation page handlers)
        :param axis: axis to perform mean function.
        """
        return np.mean([result[loss_name] for result in self.results], axis=axis)

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        raise NotImplementedError()

    def finish(self) -> None:
        pass

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch


@observable(epoch_end=int)
class TrainingPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        SummaryWriterContext.increase_global_step()
        self.trainer_or_evaluator.train(tdp)

    def finish(self) -> None:
        self.notify_observers(epoch_end=self.epoch)  # type: ignore
        self.trainer_or_evaluator.loss_reporter.flush()
        self.epoch += 1


class EvaluationPageHandler(PageHandler):
    def __init__(self, trainer, evaluator, reporter):
        self.trainer = trainer
        self.evaluator = evaluator
        self.evaluation_data: Optional[EvaluationDataPage] = None
        self.reporter = reporter
        self.results: List[CpeDetails] = []

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        if not self.trainer.calc_cpe_in_training:
            return
        # TODO: Perhaps we can make an RLTrainer param to check if continuous?
        if isinstance(self.trainer, (SACTrainer, TD3Trainer)):
            # TODO: Implement CPE for continuous algos
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
        self.evaluation_data = self.evaluation_data.compute_values(  # type: ignore
            self.trainer.gamma
        )  # type: ignore
        self.evaluation_data.validate()  # type: ignore
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


class WorldModelTrainingPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        losses = self.trainer_or_evaluator.train(tdp, batch_first=True)
        self.results.append(losses)


class WorldModelRandomTrainingPageHandler(PageHandler):
    """ Train a baseline model based on randomly shuffled data """

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        assert isinstance(tdp.training_input, PreprocessedMemoryNetworkInput)
        batch_size, _, _ = tdp.training_input.next_state.float_features.size()

        tdp = PreprocessedTrainingBatch(
            training_input=PreprocessedMemoryNetworkInput(
                state=tdp.training_input.state,
                action=tdp.training_input.action,  # type: ignore
                time_diff=torch.ones_like(
                    tdp.training_input.reward[torch.randperm(batch_size)]
                ).float(),
                # shuffle the data
                next_state=tdp.training_input.next_state._replace(
                    float_features=tdp.training_input.next_state.float_features[
                        torch.randperm(batch_size)
                    ]
                ),
                reward=tdp.training_input.reward[torch.randperm(batch_size)],
                not_terminal=tdp.training_input.not_terminal[  # type: ignore
                    torch.randperm(batch_size)
                ],
                step=None,
            ),
            extras=ExtraData(),
        )
        losses = self.trainer_or_evaluator.train(tdp, batch_first=True)
        self.results.append(losses)


class WorldModelEvaluationPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        losses = self.trainer_or_evaluator.evaluate(tdp)
        self.results.append(losses)


class ImitatorPageHandler(PageHandler):
    def __init__(self, trainer, train=True):
        super().__init__(trainer)
        self.train = train

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        losses = self.trainer_or_evaluator.train(tdp, train=self.train)
        self.results.append(losses)


class RankingTrainingPageHandler(PageHandler):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.policy_gradient_loss = []
        self.baseline_loss = []

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        _, _, rl_loss, baseline_loss = self.trainer_or_evaluator.train(tdp)
        self.results.append({"pg": rl_loss, "baseline": baseline_loss})

    def finish(self):
        self.policy_gradient_loss.append(float(self.get_mean_loss(loss_name="pg")))
        self.baseline_loss.append(float(self.get_mean_loss(loss_name="baseline")))
        self.refresh_results()


class RankingEvaluationPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        self.trainer_or_evaluator.evaluate(tdp)

    def finish(self):
        eval_res = self.trainer_or_evaluator.evaluate_post_training()
        self.results.append(eval_res)


class RewardNetTrainingPageHandler(PageHandler):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.mse_loss = []

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        mse_loss = self.trainer_or_evaluator.train(tdp)
        self.results.append({"mse": mse_loss.cpu().numpy()})

    def finish(self):
        self.mse_loss.append(float(self.get_mean_loss(loss_name="mse")))
        self.refresh_results()


def get_actual_minibatch_size(batch, minibatch_size_preset):
    if isinstance(batch, (PreprocessedTrainingBatch, RawTrainingBatch)):
        batch_size = batch.batch_size()
    elif isinstance(batch, OrderedDict):
        first_key = next(iter(batch.keys()))
        batch_size = len(batch[first_key])
    else:
        raise NotImplementedError()
    return batch_size


def feed_pages(
    data_loader,
    dataset_num_rows,
    epoch,
    minibatch_size,
    use_gpu,
    page_handler,
    batch_preprocessor=None,
):
    num_rows_processed = 0
    num_rows_to_process_for_progress_tick = max(1, dataset_num_rows // 100)
    last_percent_reported = -1

    for batch in data_loader:
        if use_gpu:
            batch = batch.cuda()
        batch_size = get_actual_minibatch_size(batch, minibatch_size)
        num_rows_processed += batch_size

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

        if batch_preprocessor:
            batch = batch_preprocessor(batch)
        page_handler.handle(batch)

    page_handler.finish()
