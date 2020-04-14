#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
from reagent.ope.estimators.estimator import Estimator, EstimatorResults
from reagent.ope.estimators.slate_estimators import (
    DMEstimator,
    LogEpisode,
    LogSample,
    NDCGSlateMetric,
    SlateContext,
    SlateEstimatorInput,
    SlateItem,
    SlateItemProbabilities,
    SlateItems,
    SlateItemValues,
    SlateModel,
    SlatePolicy,
    SlateQuery,
    SlateSlots,
)
from reagent.ope.trainers.linear_trainers import (
    DecisionTreeClassifierTrainer,
    DecisionTreeTrainer,
    LassoTrainer,
    LogisticRegressionTrainer,
    SGDClassifierTrainer,
    Trainer,
    TrainingData,
)
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
                [r[0] for v in self._dict.values() for r in v], device=self._device
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


def train(trainer: Trainer, train_dataset: MSLRDatasets, vali_dataset: MSLRDatasets):
    logging.info("training all features...")
    st = time.process_time()
    training_data = TrainingData(
        train_dataset.features,
        train_dataset.relevances,
        train_dataset.sample_weights,
        vali_dataset.features,
        vali_dataset.relevances,
        vali_dataset.sample_weights,
    )
    trainer.train(training_data)
    logging.info(f"  training time: {time.process_time() - st}")
    trainer.save_model(
        os.path.join(train_dataset.folder, trainer.name + "_all_features.pickle")
    )

    # logging.info("scoring...")
    # score = trainer.score(
    #     vali_dataset.features, vali_dataset.relevances, vali_dataset.sample_weights
    # )
    # logging.info(f"  score: {score}")

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
        os.path.join(train_dataset.folder, trainer.name + "_anchor_url_features.pickle")
    )

    # logging.info("scoring...")
    # score = trainer.score(
    #     vali_dataset.anchor_url_features,
    #     vali_dataset.relevances,
    #     vali_dataset.sample_weights,
    # )
    # logging.info(f"  score: {score}")

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
        os.path.join(train_dataset.folder, trainer.name + "_body_features.pickle")
    )

    # logging.info("scoring...")
    # score = trainer.score(
    #     vali_dataset.body_features, vali_dataset.relevances, vali_dataset.sample_weights
    # )
    # logging.info(f"  score: {score}")


def load_dataset(
    params, num_columns, anchor_url_features, body_features
) -> MSLRDatasets:
    logging.info(f"loading {params['source_file']}")
    dataset = MSLRDatasets(params, num_columns, anchor_url_features, body_features)
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


def train_all(train_dataset, vali_dataset):
    train(DecisionTreeClassifierTrainer(), train_dataset, vali_dataset)
    train(DecisionTreeTrainer(), train_dataset, vali_dataset)
    train(LassoTrainer(), train_dataset, vali_dataset)
    train(LogisticRegressionTrainer(), train_dataset, vali_dataset)
    train(SGDClassifierTrainer(), train_dataset, vali_dataset)


class TrainedModel(SlateModel):
    def __init__(self, relevances: Tensor, device=None):
        self._relevances = relevances
        self._device = device

    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        qv = context.query.value
        item_rewards = self._relevances[qv[1] : (qv[1] + qv[2])].detach().clone()
        return SlateItemValues(item_rewards)

    # def item_rewards(self, context: SlateContext) -> SlateItemValues:
    #     qv = context.query.value
    #     item_rewards = self._relevances[qv[1] : (qv[1] + qv[2])]
    #     return SlateItemValues(item_rewards)


class GroundTruthModel(SlateModel):
    def __init__(self, relevances: Tensor, device=None):
        self._relevances = relevances
        self._device = device

    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        qv = context.query.value
        doc_rewards = self._relevances[qv[1] : (qv[1] + qv[2])]
        return SlateItemValues(doc_rewards)


class MSLRPolicy(SlatePolicy):
    def __init__(
        self, relevances: Tensor, deterministic: bool, alpha: float = -1.0, device=None
    ):
        super().__init__(device)
        self._relevances = relevances
        self._deterministic = deterministic
        self._alpha = alpha

    def _item_rewards(self, context: SlateContext) -> Tensor:
        qv = context.query.value
        item_rewards = self._relevances[qv[1] : (qv[1] + qv[2])].detach().clone()
        if self._alpha >= 0:
            _, ids = torch.sort(item_rewards, descending=True)
            rank = torch.arange(1, ids.shape[0] + 1, dtype=torch.double)
            item_rewards[ids] = torch.pow(2, -1.0 * self._alpha * torch.log2(rank))
        return item_rewards

    def _query(self, context: SlateContext) -> SlateItemProbabilities:
        return SlateItemProbabilities(self._item_rewards(context), self._deterministic)


def evaluate(
    estimator: Estimator, input: SlateEstimatorInput, folder: str = "."
) -> EstimatorResults:
    logging.info(f"Evaluating {estimator}...")
    st = time.process_time()
    rs = estimator.evaluate(input)
    dt = time.process_time() - st
    print(f"Evaluating {estimator} done: {rs} in {dt}s", flush=True)
    file = os.path.join(folder, estimator.__class__.__name__ + "_results.pickle")
    try:
        with open(file, "wb") as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        logging.error(f"{file} cannot be accessed.")
    return rs


def evalute_all(
    dataset: MSLRDatasets,
    slate_size: int,
    log_trainer: Trainer,
    tgt_trainer: Trainer,
    tgt_deterministic: bool,
    num_episodes: int,
    num_samples: int,
):
    print(
        f"Run: {log_trainer.name}, {tgt_trainer.name}"
        f"[{'deterministic' if tgt_deterministic else 'stochastic'}]",
        flush=True,
    )
    logging.info("Preparing models and policies...")
    st = time.process_time()
    log_trainer.load_model(
        os.path.join(dataset.folder, log_trainer.name + "_anchor_url_features.pickle")
    )
    log_pred = log_trainer.predict(dataset.anchor_url_features)
    log_model = TrainedModel(log_pred.scores)
    log_policy = MSLRPolicy(log_pred.scores, False, 1.0)

    tgt_trainer.load_model(
        os.path.join(dataset.folder, tgt_trainer.name + "_body_features.pickle")
    )
    tgt_pred = tgt_trainer.predict(dataset.body_features)
    tgt_model = TrainedModel(tgt_pred.scores)
    tgt_policy = MSLRPolicy(tgt_pred.scores, tgt_deterministic, 1.0)

    dt = time.process_time() - st
    logging.info(f"Preparing models and policies done: {dt}s")

    logging.info("Generating log...")
    st = time.process_time()
    slots = SlateSlots(slate_size)
    queries = dataset.queries
    episodes = []
    for q in queries:
        query = SlateQuery(q)
        items = SlateItems([SlateItem(i) for i in range(q[2].item())])
        if len(items) < slate_size:
            logging.warning(
                f"Number of items ({len(items)}) less than "
                f"number of slots ({slate_size})"
            )
            continue
        context = SlateContext(query, slots, items)
        log_item_probs = log_policy(context)
        log_item_rewards = log_model.item_rewards(context)
        tgt_item_probs = tgt_policy(context)
        metric = NDCGSlateMetric(log_item_rewards)
        samples = []
        for _ in range(num_samples):
            slate = log_item_probs.sample_slate(slots)
            samples.append(LogSample(slate, slate.slot_values(log_item_rewards)))
        episodes.append(
            LogEpisode(
                context, metric, samples, None, log_item_probs, None, tgt_item_probs
            )
        )
        if len(episodes) >= num_episodes:
            break
    dt = time.process_time() - st
    logging.info(f"Generating log done: {len(episodes)} samples in {dt}s")

    input = SlateEstimatorInput(episodes, tgt_model, log_model)

    evaluate(DMEstimator(device=device), input)
    # evaluate(IPSEstimator(device=device), input)
    # evaluate(PseudoInverseEstimator(device=device), input)
    # evaluate(PBMEstimator(device=device), input)


if __name__ == "__main__":
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

    train_dataset = load_dataset(
        params["train_set"], num_columns, anchor_url_features, body_features
    )
    vali_dataset = load_dataset(
        params["vali_set"], num_columns, anchor_url_features, body_features
    )
    train_all(train_dataset, vali_dataset)

    exit(0)

    test_dataset = load_dataset(
        params["test_set"], num_columns, anchor_url_features, body_features
    )

    evalute_all(
        test_dataset, 5, DecisionTreeTrainer(), DecisionTreeTrainer(), True, 100, 100
    )
