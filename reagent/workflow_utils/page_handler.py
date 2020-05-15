#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from reagent.core.tracker import observable
from reagent.evaluation.cpe import CpeDetails
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.tensorboardX import SummaryWriterContext
from reagent.training.sac_trainer import SACTrainer
from reagent.training.td3_trainer import TD3Trainer
from reagent.types import MemoryNetworkInput, PreprocessedTrainingBatch


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


# TODO: remove.
# Use new DataLoaderWrapper & EpochIterator (see OSS train_and_evaluate_generic)
@observable(epoch_end=int)
class TrainingPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        SummaryWriterContext.increase_global_step()
        self.trainer_or_evaluator.train(tdp)

    def finish(self) -> None:
        # pyre-fixme[16]: `TrainingPageHandler` has no attribute `notify_observers`.
        self.notify_observers(epoch_end=self.epoch)
        self.trainer_or_evaluator.loss_reporter.flush()
        self.epoch += 1


# TODO: remove.
# Use new DataLoaderWrapper & EpochIterator (see OSS train_and_evaluate_generic)
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
            # pyre-fixme[16]: `Optional` has no attribute `append`.
            self.evaluation_data = self.evaluation_data.append(edp)

    def finish(self) -> None:
        if self.evaluation_data is None:
            return
        # Making sure the data is sorted for CPE
        # pyre-fixme[16]: `Optional` has no attribute `sort`.
        self.evaluation_data = self.evaluation_data.sort()
        # pyre-fixme[16]: `Optional` has no attribute `compute_values`.
        self.evaluation_data = self.evaluation_data.compute_values(self.trainer.gamma)
        # pyre-fixme[16]: `Optional` has no attribute `validate`.
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


class WorldModelTrainingPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        losses = self.trainer_or_evaluator.train(tdp)
        self.results.append(losses)


class WorldModelRandomTrainingPageHandler(PageHandler):
    """ Train a baseline model based on randomly shuffled data """

    # pyre-fixme[14]: `handle` overrides method defined in `PageHandler` inconsistently.
    def handle(self, training_input: MemoryNetworkInput) -> None:
        _, batch_size, _ = training_input.next_state.float_features.size()

        tdp = MemoryNetworkInput(
            state=training_input.state,
            action=training_input.action,
            time_diff=torch.ones_like(training_input.reward),
            # shuffle the data
            next_state=training_input.next_state._replace(
                float_features=training_input.next_state.float_features[
                    :, torch.randperm(batch_size), :
                ]
            ),
            reward=training_input.reward[:, torch.randperm(batch_size)],
            not_terminal=training_input.not_terminal[  # type: ignore
                :, torch.randperm(batch_size)
            ],
            step=None,
        )
        losses = self.trainer_or_evaluator.train(tdp)
        self.results.append(losses)


class WorldModelEvaluationPageHandler(PageHandler):
    # pyre-fixme[14]: `handle` overrides method defined in `PageHandler` inconsistently.
    def handle(self, tdp: MemoryNetworkInput) -> None:
        losses = self.trainer_or_evaluator.evaluate(tdp)
        self.results.append(losses)


class RankingTrainingPageHandler(PageHandler):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.policy_gradient_loss: List[float] = []
        self.baseline_loss: List[float] = []
        self.per_seq_probs: List[float] = []

    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        res_dict = self.trainer_or_evaluator.train(tdp)
        self.results.append(res_dict)

    def finish(self):
        if "ips_rl_loss" in self.results[0]:
            self.policy_gradient_loss.append(
                float(self.get_mean_loss(loss_name="ips_rl_loss"))
            )
        if "baseline_loss" in self.results[0]:
            self.baseline_loss.append(
                float(self.get_mean_loss(loss_name="baseline_loss"))
            )
        if "per_seq_probs" in self.results[0]:
            self.per_seq_probs.append(
                float(self.get_mean_loss(loss_name="per_seq_probs"))
            )
        self.refresh_results()


@observable(epoch_end=int)
class RankingEvaluationPageHandler(PageHandler):
    def handle(self, tdp: PreprocessedTrainingBatch) -> None:
        self.trainer_or_evaluator.evaluate(tdp)

    def finish(self):
        self.notify_observers(epoch_end=self.epoch)  # type: ignore
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


# TODO: remove.
# Use new DataLoaderWrapper & EpochIterator (see OSS train_and_evaluate_generic)
def get_actual_minibatch_size(batch, minibatch_size_preset):
    try:
        return batch.batch_size()
    except AttributeError:
        pass
    if isinstance(batch, OrderedDict):
        first_key = next(iter(batch.keys()))
        batch_size = len(batch[first_key])
    else:
        raise NotImplementedError()
    return batch_size


# TODO: remove.
# Use new DataLoaderWrapper & EpochIterator (see OSS train_and_evaluate_generic)
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
