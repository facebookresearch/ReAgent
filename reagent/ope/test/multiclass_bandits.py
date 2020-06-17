#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from reagent.ope.estimators.contextual_bandits_estimators import (
    Action,
    ActionDistribution,
    ActionRewards,
    BanditsEstimatorInput,
    BanditsModel,
    DMEstimator,
    DoublyRobustEstimator,
    IPSEstimator,
    LogSample,
)
from reagent.ope.estimators.estimator import Estimator, Evaluator
from reagent.ope.estimators.types import ActionSpace, Policy, Trainer, TrainingData
from reagent.ope.trainers.linear_trainers import (
    DecisionTreeTrainer,
    LogisticRegressionTrainer,
    SGDClassifierTrainer,
)
from torch import Tensor


@dataclass(frozen=True)
class MultiClassDataRow(object):
    feature: torch.Tensor
    label: torch.Tensor
    one_hot: torch.Tensor


class UCIMultiClassDataset:
    """
    To load and hold UCI classification datasets:
    https://archive.ics.uci.edu/ml/datasets.php?task=cla&sort=nameUp&view=table
    Also to convert it to contextual bandits problems
    References: https://arxiv.org/abs/1103.4601
    """

    def __init__(self, params, device=None):
        if "file" not in params:
            raise Exception('Please define "file" in "dataset"')
        if "label_col" not in params:
            raise Exception('Please define "label_col" in "dataset"')

        index_col = params["index_col"] if "index_col" in params else None
        label_col = params["label_col"]
        sep = params["sep"] if "sep" in params else ","
        self._config_file = params["file"]
        self._data_frame = pd.read_csv(
            self._config_file,
            sep=sep,
            header=None,
            index_col=index_col if index_col is not None else False,
        )
        file_col_len = self._data_frame.shape[1] + (1 if index_col is not None else 0)
        if label_col < 0:
            label_col = file_col_len + label_col
        frame_label_col = label_col
        if index_col is not None and label_col > index_col:
            label_col = label_col - 1
        self._features = torch.as_tensor(
            self._data_frame.iloc[
                :, [i for i in range(self._data_frame.shape[1]) if i != label_col]
            ].values,
            dtype=torch.float32,
            device=device,
        )
        self._classes = self._data_frame[frame_label_col].unique()
        self._classes.sort()
        self._labels = self._data_frame[frame_label_col].values
        self._class_indices = torch.tensor(
            [np.where(self._classes == i)[0][0] for i in self._labels],
            dtype=torch.long,
            device=device,
        )
        self._one_hots = torch.zeros(
            (self._class_indices.shape[0], len(self._classes)),
            dtype=torch.int,
            device=device,
        )
        self._one_hots[
            torch.arange(self._one_hots.shape[0], dtype=torch.long), self._class_indices
        ] = 1

        self.device = device

    def __len__(self):
        return len(self._data_frame)

    def __getitem__(self, idx) -> MultiClassDataRow:
        return MultiClassDataRow(
            self._features[idx], self._class_indices[idx], self._one_hots[idx]
        )

    @property
    def config_file(self) -> str:
        return self._config_file

    @property
    def num_features(self) -> int:
        return self._features.shape[1]

    @property
    def num_actions(self) -> int:
        return len(self._classes)

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def labels(self) -> torch.Tensor:
        return self._class_indices

    @property
    def one_hots(self) -> torch.Tensor:
        return self._one_hots

    def train_val_test_split(
        self, ratios: Tuple[float, float] = (0.8, 0.8), device=None
    ):
        total_len = len(self._data_frame)
        train_len = int(total_len * ratios[0])
        train_choices = random.sample(range(total_len), train_len)
        train_x = np.take(self._features, train_choices, axis=0)
        train_y = np.take(self._class_indices, train_choices)
        train_r = np.take(self._one_hots, train_choices, axis=0)
        fit_len = int(train_len * ratios[1])
        fit_choices = random.sample(range(train_len), fit_len)
        fit_x = np.take(train_x, fit_choices, axis=0)
        fit_y = np.take(train_y, fit_choices)
        fit_r = np.take(train_r, fit_choices, axis=0)
        val_x = np.delete(train_x, fit_choices, axis=0)
        val_y = np.delete(train_y, fit_choices)
        val_r = np.delete(train_r, fit_choices, axis=0)
        test_x = np.delete(self._features, train_choices, axis=0)
        test_y = np.delete(self._class_indices, train_choices)
        test_r = np.delete(self._one_hots, train_choices, axis=0)
        return (
            torch.as_tensor(fit_x, dtype=torch.float, device=device),
            torch.as_tensor(fit_y, dtype=torch.float, device=device),
            torch.as_tensor(fit_r, dtype=torch.float, device=device),
            torch.as_tensor(val_x, dtype=torch.float, device=device),
            torch.as_tensor(val_y, dtype=torch.float, device=device),
            torch.as_tensor(val_r, dtype=torch.float, device=device),
            torch.as_tensor(test_x, dtype=torch.float, device=device),
            torch.as_tensor(test_y, dtype=torch.float, device=device),
            torch.as_tensor(test_r, dtype=torch.float, device=device),
        )


@dataclass(frozen=True)
class MultiClassContext:
    query_id: int


class MultiClassModel(BanditsModel):
    def __init__(self, features: Tensor, rewards: Tensor):
        self._features = features
        self._rewards = rewards

    def _action_rewards(self, context: MultiClassContext) -> ActionRewards:
        return ActionRewards(self._rewards[context.query_id])


class MultiClassPolicy(Policy):
    def __init__(
        self,
        action_space: ActionSpace,
        action_distributions: Tensor,
        epsilon: float,
        device=None,
    ):
        super().__init__(action_space, device)
        self._action_ditributions = action_distributions
        self._exploitation_prob = 1.0 - epsilon
        self._exploration_prob = epsilon / len(self.action_space)

    def _query(self, query_id: int) -> Tuple[Action, ActionDistribution]:
        dist = self._action_ditributions[query_id]
        dist = dist * self._exploitation_prob + self._exploration_prob
        action = torch.multinomial(dist, 1).item()
        return Action(action), ActionDistribution(dist)


def evaluate_all(
    experiments: Iterable[Tuple[Iterable[Estimator], int]],
    dataset: UCIMultiClassDataset,
    log_trainer: Trainer,
    log_epsilon: float,
    tgt_trainer: Trainer,
    tgt_epsilon: float,
    max_num_workers: int,
    device=None,
):
    action_space = ActionSpace(dataset.num_actions)
    config_path = PurePath(dataset.config_file)
    data_name = config_path.stem
    log_model_name = data_name + "_" + log_trainer.__class__.__name__ + ".pickle"
    log_model_file = str(config_path.with_name(log_model_name))
    tgt_model_name = data_name + "_" + tgt_trainer.__class__.__name__ + ".pickle"
    tgt_model_file = str(config_path.with_name(tgt_model_name))

    log_trainer.load_model(log_model_file)
    tgt_trainer.load_model(tgt_model_file)
    if not log_trainer.is_trained or not tgt_trainer.is_trained:
        (
            train_x,
            train_y,
            train_r,
            val_x,
            val_y,
            val_r,
            test_x,
            test_y,
            test_r,
        ) = dataset.train_val_test_split((0.8, 0.8))
        trainer_data = TrainingData(train_x, train_y, None, val_x, val_y, None)
        if not log_trainer.is_trained:
            log_trainer.train(trainer_data)
            log_trainer.save_model(log_model_file)
        if not tgt_trainer.is_trained:
            tgt_trainer.train(trainer_data)
            tgt_trainer.save_model(tgt_model_file)

    log_results = log_trainer.predict(dataset.features)
    log_policy = MultiClassPolicy(action_space, log_results.probabilities, log_epsilon)

    tgt_results = tgt_trainer.predict(dataset.features)
    tgt_policy = MultiClassPolicy(action_space, tgt_results.probabilities, tgt_epsilon)

    inputs = []
    tasks = []
    total_queries = len(dataset)
    for estimators, num_samples in experiments:
        samples = []
        for i in range(num_samples):
            qid = random.randrange(total_queries)
            label = int(dataset.labels[qid].item())
            log_action, log_action_probabilities = log_policy(qid)
            log_reward = 1.0 if log_action.value == label else 0.0
            tgt_action, tgt_action_probabilities = tgt_policy(qid)
            ground_truth_reward = 1.0 if tgt_action.value == label else 0.0
            item_feature = dataset.features[qid]
            samples.append(
                LogSample(
                    qid,
                    log_action,
                    log_reward,
                    log_action_probabilities,
                    tgt_action_probabilities,
                    ground_truth_reward,
                    item_feature,
                )
            )
        tasks.append((estimators, BanditsEstimatorInput(action_space, samples)))

    logging.info("start evaluating...")
    st = time.perf_counter()
    evaluator = Evaluator(tasks, max_num_workers)
    results = evaluator.evaluate()
    Evaluator.report_results(results)
    logging.info(f"evaluating done in {time.perf_counter() - st}s")
    return results


DEFAULT_ITERATIONS = 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info(f"working dir - {os.getcwd()}")

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-p", "--parameters", help="Path to config file.")
    args = parser.parse_args(sys.argv[1:])

    with open(args.parameters, "r") as f:
        params = json.load(f)

    if "dataset" not in params:
        raise Exception('Please define "dataset" in config file')

    random.seed(1234)
    np.random.seed(1234)
    torch.random.manual_seed(1234)

    dataset = UCIMultiClassDataset(params["dataset"])
    log_trainer = LogisticRegressionTrainer()
    log_epsilon = 0.1
    tgt_trainer = SGDClassifierTrainer()
    tgt_epsilon = 0.1
    dm_trainer = DecisionTreeTrainer()
    experiments = [
        (
            (
                DMEstimator(DecisionTreeTrainer()),
                IPSEstimator(),
                DoublyRobustEstimator(DecisionTreeTrainer()),
            ),
            1000,
        )
        for _ in range(100)
    ]
    evaluate_all(
        experiments, dataset, log_trainer, log_epsilon, tgt_trainer, tgt_epsilon, 0
    )
