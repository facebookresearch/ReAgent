#!/usr/bin/env python3

import argparse
import itertools
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import OrderedDict
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from reagent.ope.estimators.estimator import Evaluator
from reagent.ope.estimators.slate_estimators import (
    DCGSlateMetric,
    DMEstimator,
    DoublyRobustEstimator,
    ERRSlateMetric,
    IPSEstimator,
    LogSample,
    NDCGSlateMetric,
    PassThruDistribution,
    PBMEstimator,
    PseudoInverseEstimator,
    RankingDistribution,
    RewardDistribution,
    SlateContext,
    SlateEstimator,
    SlateEstimatorInput,
    SlateItemFeatures,
    SlateItemValues,
    SlateModel,
    SlateQuery,
    SlateSlots,
)
from reagent.ope.estimators.types import Trainer, TrainingData
from reagent.ope.trainers.linear_trainers import DecisionTreeTrainer, LassoTrainer
from reagent.ope.utils import Clamper
from torch import Tensor


# Slate test using Microsoft Learning to Rank Datasets (MSLR-WEB30K):
#   https://www.microsoft.com/en-us/research/project/mslr/


class MSLRDatasets:
    def __init__(
        self,
        params,
        num_columns: int,
        anchor_url_features: List[int],
        body_features: List[int],
        dataset_name: str = "",
        device=None,
    ):
        if "folder" not in params:
            raise Exception('Please define "folder" in "dataset"')
        self._folder = params["folder"]
        self._source_file = (
            params["source_file"] if "source_file" in params else ["train.txt"]
        )
        self._cache_file = params["cache_file"] if "cache_file" in params else ""
        self._num_columns = num_columns
        self._anchor_url_features = anchor_url_features
        self._body_features = body_features

        self._device = device

        self._dict = None
        self._queries = None
        self._features = None
        self._relevances = None
        self._sample_weights = None

        self._train_data = None
        self._validation_data = None
        self._test_data = None

        self._name = dataset_name

    @property
    def name(self) -> str:
        return self._name

    def _add(self, qid: Optional[int], feature_list: List[Tuple[float, Tensor]]):
        if qid is None or len(feature_list) == 0:
            return
        if qid in self._dict:
            self._dict[qid].extend(feature_list)
        else:
            self._dict[qid] = feature_list

    def load(self):
        pickle_file = os.path.join(self._folder, self._cache_file)
        if len(self._cache_file) > 0 and os.access(pickle_file, os.R_OK):
            logging.info(f"loading {pickle_file}")
            with open(pickle_file, "rb") as f:
                self._queries, self._features, self._relevances = pickle.load(f)
                self._cache_file = ""
            del f
        else:
            self._dict = OrderedDict()
            text_file = os.path.join(self._folder, self._source_file)
            logging.info(f"loading {text_file}")
            if not os.access(text_file, os.R_OK):
                logging.warning(f"{text_file} cannot be accessed.")
                return
            with open(text_file, "r") as f:
                c = 0
                st = time.process_time()
                # feature index starts with 1, so leave features[0] as padding
                features = list(range(self._num_columns - 1))
                features_list = []
                prev_qid = None
                for line in f.readlines():
                    tokens = line.strip().split()
                    if len(tokens) != self._num_columns:
                        continue
                    rel = int(tokens[0])
                    qid = int(tokens[1].split(":")[1])
                    for i in range(2, self._num_columns):
                        feature = tokens[i].split(":")
                        features[i - 1] = float(feature[1])
                    f_tensor = torch.tensor(features, device=self._device)
                    if prev_qid is None:
                        prev_qid = qid
                        features_list.append((rel, f_tensor))
                    elif prev_qid != qid:
                        self._add(prev_qid, features_list)
                        prev_qid = qid
                        features_list = []
                        features_list.append((rel, f_tensor))
                    else:
                        features_list.append((rel, f_tensor))
                    c += 1
                    if c % 100000 == 0:
                        print(f"{c} - {(time.process_time() - st) / c}")
                self._add(prev_qid, features_list)

    def save(self):
        if len(self._cache_file) == 0 or self._dict is None:
            return
        pickle_file = os.path.join(self._folder, self._cache_file)
        try:
            with open(pickle_file, "wb") as f:
                self._load_features()
                pickle.dump(
                    (self.queries, self._features, self.relevances),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            logging.error(f"{pickle_file} cannot be accessed.")

    @property
    def queries(self) -> Tensor:
        if self._queries is None:
            rows = []
            c = 0
            for i in self._dict.items():
                s = len(i[1])
                rows.append([i[0], c, s])
                c += s
            self._queries = torch.tensor(rows, dtype=torch.int, device=self._device)
        return self._queries

    def _load_features(self):
        if self._features is None:
            self._features = torch.stack([r[1] for v in self._dict.values() for r in v])

    @property
    def features(self) -> Tensor:
        self._load_features()
        return self._features[:, 1:]

    @property
    def all_features(self) -> Tensor:
        return self.features

    @property
    def anchor_url_features(self) -> Tensor:
        self._load_features()
        return (
            self._features[:, self._anchor_url_features]
            if self._anchor_url_features is not None
            else None
        )

    @property
    def body_features(self) -> Tensor:
        self._load_features()
        return (
            self._features[:, self._body_features]
            if self._body_features is not None
            else None
        )

    @property
    def relevances(self) -> Tensor:
        if self._relevances is None:
            self._relevances = torch.tensor(
                [r[0] for r in itertools.chain(self._dict.values())],
                device=self._device,
            )
        return self._relevances

    @property
    def sample_weights(self) -> Tensor:
        if self._sample_weights is None:
            samples = self.queries[:, 2]
            self._sample_weights = torch.repeat_interleave(
                samples.to(dtype=torch.float).reciprocal(), samples.to(dtype=torch.long)
            )
        return self._sample_weights

    @property
    def folder(self) -> str:
        return self._folder

    @property
    def source_file(self) -> List[str]:
        return self._source_file

    @property
    def cache_file(self) -> str:
        return self._cache_file


def train(
    trainer: Trainer,
    train_dataset: MSLRDatasets,
    vali_dataset: MSLRDatasets,
    prefix: str = "",
):
    logging.info("training all features...")
    st = time.process_time()
    training_data = TrainingData(
        train_dataset.all_features,
        train_dataset.relevances,
        train_dataset.sample_weights,
        vali_dataset.all_features,
        vali_dataset.relevances,
        vali_dataset.sample_weights,
    )
    trainer.train(training_data)
    logging.info(f"  training time: {time.process_time() - st}")
    trainer.save_model(
        os.path.join(
            train_dataset.folder, trainer.name + "_" + prefix + "_all_features.pickle"
        )
    )

    logging.info("scoring...")
    score = trainer.score(
        vali_dataset.all_features, vali_dataset.relevances, vali_dataset.sample_weights
    )
    logging.info(f"  score: {score}")

    logging.info("training anchor_url features...")
    st = time.process_time()
    trainer.train(
        TrainingData(
            train_dataset.anchor_url_features,
            train_dataset.relevances,
            train_dataset.sample_weights,
            vali_dataset.anchor_url_features,
            vali_dataset.relevances,
            vali_dataset.sample_weights,
        )
    )
    logging.info(f"  training time: {time.process_time() - st}")
    trainer.save_model(
        os.path.join(
            train_dataset.folder,
            trainer.name + "_" + prefix + "_anchor_url_features.pickle",
        )
    )

    logging.info("scoring...")
    score = trainer.score(
        vali_dataset.anchor_url_features,
        vali_dataset.relevances,
        vali_dataset.sample_weights,
    )
    logging.info(f"  score: {score}")

    logging.info("training body features...")
    st = time.process_time()
    trainer.train(
        TrainingData(
            train_dataset.body_features,
            train_dataset.relevances,
            train_dataset.sample_weights,
            vali_dataset.body_features,
            vali_dataset.relevances,
            vali_dataset.sample_weights,
        )
    )
    logging.info(f"  training time: {time.process_time() - st}")
    trainer.save_model(
        os.path.join(
            train_dataset.folder, trainer.name + "_" + prefix + "_body_features.pickle"
        )
    )

    logging.info("scoring...")
    score = trainer.score(
        vali_dataset.body_features, vali_dataset.relevances, vali_dataset.sample_weights
    )
    logging.info(f"  score: {score}")


def load_dataset(
    params, num_columns, anchor_url_features, body_features, dataset_name=""
) -> MSLRDatasets:
    logging.info(f"loading {params['source_file']}")
    dataset = MSLRDatasets(
        params, num_columns, anchor_url_features, body_features, dataset_name
    )
    st = time.process_time()
    dataset.load()
    logging.info(f"  load time: {time.process_time() - st}")

    st = time.process_time()
    dataset.save()
    logging.info(f"  save time: {time.process_time() - st}")
    logging.info(
        f"  queries: {dataset.queries.shape}"
        f", features: {dataset.features.shape}"
        f", sample_weights: {dataset.sample_weights.shape}"
        f", relevance: {dataset.relevances.shape}"
        f", anchor_url: {dataset.anchor_url_features.shape}"
        f", body: {dataset.body_features.shape}"
    )
    return dataset


def train_all(train_dataset, vali_dataset, prefix: str = ""):
    # train(DecisionTreeClassifierTrainer(), train_dataset, vali_dataset)
    train(DecisionTreeTrainer(), train_dataset, vali_dataset, prefix)
    train(LassoTrainer(), train_dataset, vali_dataset, prefix)
    # train(LogisticRegressionTrainer(), train_dataset, vali_dataset)
    # train(SGDClassifierTrainer(), train_dataset, vali_dataset)


def train_models(params):
    all_dataset = load_dataset(
        params["all_set"], num_columns, anchor_url_features, body_features
    )
    half_dataset = load_dataset(
        params["first_set"], num_columns, anchor_url_features, body_features
    )
    vali_dataset = load_dataset(
        params["vali_set"], num_columns, anchor_url_features, body_features
    )
    train_all(all_dataset, vali_dataset, "all")
    train_all(half_dataset, vali_dataset, "half")


class MSLRModel(SlateModel):
    def __init__(self, relevances: Tensor, device=None):
        self._relevances = relevances
        self._device = device

    def item_relevances(self, context: SlateContext) -> Tensor:
        qv = context.query.value
        if context.params is None:
            relevances = self._relevances[qv[1] : (qv[1] + qv[2])].detach().clone()
        else:
            relevances = (
                self._relevances[qv[1] : (qv[1] + qv[2])][context.params]
                .detach()
                .clone()
            )
        return relevances

    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        return SlateItemValues(self.item_relevances(context))


def evaluate(
    experiments: Iterable[Tuple[Iterable[SlateEstimator], int]],
    dataset: MSLRDatasets,
    slate_size: int,
    item_size: int,
    metric_func: str,
    log_trainer: Trainer,
    log_distribution: RewardDistribution,
    log_features: str,
    tgt_trainer: Trainer,
    tgt_distribution: RewardDistribution,
    tgt_features: str,
    dm_features: str,
    max_num_workers: int,
    device=None,
):
    assert slate_size < item_size
    print(
        f"Evaluate All:"
        f" slate_size={slate_size}, item_size={item_size}, metric={metric_func}"
        f", Log=[{log_trainer.name}, {log_distribution}, {log_features}]"
        f", Target=[{tgt_trainer.name}, {tgt_distribution}, {tgt_features}]"
        f", DM=[{dm_features}]"
        f", Workers={max_num_workers}, device={device}",
        flush=True,
    )
    logging.info("Preparing models and policies...")
    st = time.perf_counter()
    log_trainer.load_model(
        os.path.join(
            dataset.folder, log_trainer.name + "_all_" + log_features + ".pickle"
        )
    )
    # calculate behavior model scores
    log_pred = log_trainer.predict(getattr(dataset, log_features))

    tgt_trainer.load_model(
        os.path.join(
            dataset.folder, tgt_trainer.name + "_all_" + tgt_features + ".pickle"
        )
    )
    # calculate target model scores
    tgt_pred = tgt_trainer.predict(getattr(dataset, tgt_features))

    dm_train_features = getattr(dataset, dm_features)

    slots = SlateSlots(slate_size)

    dt = time.perf_counter() - st
    logging.info(f"Preparing models and policies done: {dt}s")

    total_samples = 0
    for _, num_samples in experiments:
        total_samples += num_samples
    logging.info(f"Generating log: total_samples={total_samples}")
    st = time.perf_counter()
    tasks = []
    samples_generated = 0
    total_queries = dataset.queries.shape[0]
    for estimators, num_samples in experiments:
        samples = []
        for _ in range(num_samples):
            # randomly sample a query
            q = dataset.queries[random.randrange(total_queries)]
            doc_size = int(q[2])
            if doc_size < item_size:
                # skip if number of docs is less than item_size
                continue
            si = int(q[1])
            ei = si + doc_size
            # using top item_size docs for logging
            log_scores, item_choices = log_pred.scores[si:ei].sort(
                dim=0, descending=True
            )
            log_scores = log_scores[:item_size]
            item_choices = item_choices[:item_size]
            log_item_probs = log_distribution(SlateItemValues(log_scores))
            tgt_scores = tgt_pred.scores[si:ei][item_choices].detach().clone()
            tgt_item_probs = tgt_distribution(SlateItemValues(tgt_scores))
            tgt_slot_expectation = tgt_item_probs.slot_item_expectations(slots)
            gt_item_rewards = SlateItemValues(dataset.relevances[si:ei][item_choices])
            gt_rewards = tgt_slot_expectation.expected_rewards(gt_item_rewards)
            if metric_func == "dcg":
                metric = DCGSlateMetric(device=device)
            elif metric_func == "err":
                metric = ERRSlateMetric(4.0, device=device)
            else:
                metric = NDCGSlateMetric(gt_item_rewards, device=device)
            query = SlateQuery((si, ei))
            context = SlateContext(query, slots, item_choices)
            slot_weights = metric.slot_weights(slots)
            gt_reward = metric.calculate_reward(slots, gt_rewards, None, slot_weights)
            if tgt_item_probs.is_deterministic:
                tgt_slate_prob = 1.0
                log_slate = tgt_item_probs.sample_slate(slots)
                log_reward = gt_reward
            else:
                tgt_slate_prob = float("nan")
                log_slate = log_item_probs.sample_slate(slots)
                log_rewards = log_slate.slot_values(gt_item_rewards)
                log_reward = metric.calculate_reward(
                    slots, log_rewards, None, slot_weights
                )
            log_slate_prob = log_item_probs.slate_probability(log_slate)
            item_features = SlateItemFeatures(dm_train_features[si:ei][item_choices])
            sample = LogSample(
                context,
                metric,
                log_slate,
                log_reward,
                log_slate_prob,
                None,
                log_item_probs,
                tgt_slate_prob,
                None,
                tgt_item_probs,
                gt_reward,
                slot_weights,
                None,
                item_features,
            )
            samples.append(sample)
            samples_generated += 1
            if samples_generated % 1000 == 0:
                logging.info(
                    f"  samples generated: {samples_generated}, {100 * samples_generated / total_samples:.1f}%"
                )
        tasks.append((estimators, SlateEstimatorInput(samples)))
    dt = time.perf_counter() - st
    logging.info(f"Generating log done: {total_samples} samples in {dt}s")

    logging.info("start evaluating...")
    st = time.perf_counter()
    evaluator = Evaluator(tasks, max_num_workers)
    Evaluator.report_results(evaluator.evaluate())
    logging.info(f"evaluating done in {time.perf_counter() - st}s")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    logging.basicConfig(
        format="%(asctime)-15s_%(levelname)s: %(message)s", level=logging.INFO
    )

    logging.info(f"working dir - {os.getcwd()}")

    random.seed(1234)
    np.random.seed(1234)
    torch.random.manual_seed(1234)

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-p", "--parameters", help="Path to config file.")
    args = parser.parse_args(sys.argv[1:])

    with open(args.parameters, "r") as f:
        params = json.load(f)

    if "train_set" not in params:
        logging.error('"train_set" not defined')
        exit(1)

    if "vali_set" not in params:
        logging.error('"vali_set" not defined')
        exit(1)

    if "test_set" not in params:
        logging.error('"test_set" not defined')
        exit(1)

    # device = torch.device("cuda") if torch.cuda.is_available() else None
    device = None

    num_columns = params["num_columns"] if "num_columns" in params else 138
    anchor_url_features = (
        params["anchor_url_features"] if "anchor_url_features" in params else None
    )
    body_features = params["body_features"] if "body_features" in params else None

    # uncomment to train behavior and target models
    # train_models(params)

    test_dataset = load_dataset(
        params["second_set"],
        num_columns,
        anchor_url_features,
        body_features,
        "second_set",
    )
    weight_clamper = Clamper(min_v=0.0)
    estimators = [
        DMEstimator(DecisionTreeTrainer(), 0.5, device=device),
        IPSEstimator(weight_clamper=weight_clamper, device=device),
        DoublyRobustEstimator(
            DecisionTreeTrainer(), 0.5, weight_clamper, False, device
        ),
        DoublyRobustEstimator(DecisionTreeTrainer(), 0.5, weight_clamper, True, device),
        PseudoInverseEstimator(weight_clamper=weight_clamper, device=device),
        PBMEstimator(weight_clamper=weight_clamper, device=device),
    ]

    metrics = ["ndcg", "err"]
    alphas = [0.0, 1.0, 2.0]
    trainers = [
        (DecisionTreeTrainer(), LassoTrainer()),
        (LassoTrainer(), DecisionTreeTrainer()),
    ]
    for log_trainer, tgt_trainers in trainers:
        for metric in metrics:
            for alpha in alphas:
                evaluate(
                    [(estimators, 200)] * 4,
                    test_dataset,
                    5,
                    20,
                    metric,
                    log_trainer,
                    RankingDistribution(alpha),
                    "anchor_url_features",
                    tgt_trainers,
                    PassThruDistribution(),
                    "body_features",
                    "all_features",
                    4,
                )
