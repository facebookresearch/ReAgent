#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from typing import (
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.multiprocessing as mp
from reagent.ope.estimators.estimator import Estimator, Evaluator
from reagent.ope.estimators.slate_estimators import (
    DCGSlateMetric,
    ERRSlateMetric,
    FrechetDistribution,
    IPSEstimator,
    LogSample,
    NDCGSlateMetric,
    PBMEstimator,
    PseudoInverseEstimator,
    RankingDistribution,
    RewardDistribution,
    SlateContext,
    SlateEstimator,
    SlateEstimatorInput,
    SlateItemValues,
    SlateModel,
    SlateQuery,
    SlateSlots,
    SlateSlotValues,
)
from reagent.ope.utils import RunningAverage


# Slate test using Yandex Personalized Web Search Dataset:
#   https://www.kaggle.com/c/yandex-personalized-web-search-challenge/

RELEVANT_THRESHOLD = 49
HIGHLY_RELEVANT_THRESHOLD = 399
MAX_SLATE_SIZE = 10
MIN_QUERY_COUNT = 10


def click_to_relevances(
    clicks: Iterable[Tuple[int, int]], urls: Sequence[Tuple[int, int]]
) -> Tuple[List[float], Mapping[Tuple[int, int], float]]:
    position_relevances = [0.0] * max(len(urls), MAX_SLATE_SIZE)
    url_relevances = {url: 0.0 for url in urls}
    for i, dt in clicks:
        r = 0.0
        if dt > HIGHLY_RELEVANT_THRESHOLD:
            r = 2.0
        elif dt > RELEVANT_THRESHOLD:
            r = 1.0
        position_relevances[i] = r
        url_relevances[urls[i]] = r
    return position_relevances, url_relevances


class LoggedQuery:
    def __init__(
        self,
        user_id: int,
        query_id: int,
        query_terms: Tuple[int],
        list: Sequence[Tuple[int, int]],
    ):
        self._user_id = user_id
        self._query_id = query_id
        self._query_terms = query_terms
        self._list = list
        self._clicks: List[Tuple[int, int]] = []
        self._position_relevances: Optional[List[float]] = None
        self._url_relevances: Optional[MutableMapping[Tuple[int, int], float]] = None

    def click(self, url_id: int, dwell_time: int):
        self._position_relevances = None
        self._url_relevances = None
        i = 0
        for r in self.list:
            if url_id == r[0]:
                self.clicks.append((i, dwell_time))
                break
            i += 1

    @property
    def user_id(self):
        return self._user_id

    @property
    def query_id(self):
        return self._query_id

    @property
    def query_terms(self):
        return self._query_terms

    @property
    def list(self):
        return self._list

    @property
    def clicks(self):
        return self._clicks

    def _click_to_relevances(self):
        self._position_relevances = [0.0] * max(len(self._list), MAX_SLATE_SIZE)
        self._url_relevances = {url: 0.0 for url in self._list}
        for i, dt in self.clicks:
            r = 0.0
            if dt > HIGHLY_RELEVANT_THRESHOLD:
                r = 2.0
            elif dt > RELEVANT_THRESHOLD:
                r = 1.0
            self._position_relevances[i] = r
            self._url_relevances[self._list[i]] = r

    @property
    def position_relevances(self):
        if self._position_relevances is None:
            self._click_to_relevances()
        return self._position_relevances

    @property
    def url_relevances(self):
        if self._url_relevances is None:
            self._click_to_relevances()
        return self._url_relevances


class TrainingQuery:
    def __init__(self, query_id: int, query_terms: Tuple[int]):
        self._query_id = query_id
        self._query_terms = query_terms
        self._count = 0
        self._url_relevances: Union[
            Sequence[Tuple[Tuple[int, int], float]],
            MutableMapping[Tuple[int, int], float],
        ] = {}
        self._position_relevances = [RunningAverage() for _ in range(MAX_SLATE_SIZE)]

    def add(self, query: LoggedQuery):
        self._count += 1
        urs = query.url_relevances
        for item_id, r in urs.items():
            if item_id not in self._url_relevances:
                self._url_relevances[item_id] = RunningAverage(r)
            else:
                self._url_relevances[item_id].add(r)
        prs = query.position_relevances
        for i in range(MAX_SLATE_SIZE):
            self._position_relevances[i].add(prs[i])

    def merge(self, other: "TrainingQuery"):
        for i, r in other.url_relevances.items():
            if i not in self._url_relevances:
                self._url_relevances[i] = RunningAverage(r)
            else:
                self._url_relevances[i].add(r)
        for i in range(MAX_SLATE_SIZE):
            self._position_relevances[i].add(other.position_relevances[i])

    def finalize(self):
        self._url_relevances = {k: v.average for k, v in self._url_relevances.items()}
        self._position_relevances = [v.average for v in self._position_relevances]

    def pack(self):
        if isinstance(self._url_relevances, Mapping):
            self._url_relevances = list(self._url_relevances.items())

    def _unpack(self):
        if isinstance(self._url_relevances, Sequence):
            self._url_relevances = {v[0]: v[1] for v in self._url_relevances}

    @property
    def count(self):
        return self._count

    @property
    def query_id(self):
        return self._query_id

    @property
    def query_terms(self):
        return self._query_terms

    @property
    def url_relevances(self):
        self._unpack()
        return self._url_relevances

    @property
    def position_relevances(self):
        return self._position_relevances


def create_cache(params):
    if "folder" not in params:
        raise Exception('Please define "folder" in "raw_data"')
    folder = params["folder"] if "folder" in params else ""
    if len(folder) == 0:
        folder = os.getcwd()
    cache_folder = params["cache_folder"] if "cache_folder" in params else folder
    if len(cache_folder) == 0:
        cache_folder = folder
    source_file = params["source_file"] if "source_file" in params else ""
    if len(source_file) == 0:
        raise Exception('"source_file" not defined!')
    total_days = params["total_days"] if "total_days" in params else 27
    text_file = os.path.join(folder, source_file)
    logging.info(f"loading {text_file}")
    if not os.access(text_file, os.R_OK):
        logging.warning(f"{text_file} cannot be accessed.")
        return
    for d in range(1, total_days + 1):
        pickle_file = os.path.join(cache_folder, f"{source_file}_{d:02}.pickle")
        logging.info(f"creating cache for day {d:02}: {pickle_file}")
        queries = []
        st = time.process_time()
        with open(text_file, "r") as f:
            curr_sess = None
            curr_user = -1
            num_sess = 0
            last_click = None
            for line in f:
                tokens = line.strip().split()
                tlen = len(tokens)
                if tlen == 4 and tokens[1] == "M":
                    if last_click is not None:
                        query = curr_sess[2][last_click[0]]
                        query.click(last_click[1], 10000)
                        last_click = None
                    day = int(tokens[2])
                    if day != d:
                        continue
                    num_sess += 1
                    if num_sess % 100000 == 0:
                        logging.info(f"  {num_sess} session processed...")
                    if curr_sess is not None:
                        qids = set()
                        for q in curr_sess[2].values():
                            if len(q.clicks) > 0:
                                queries.append(q)
                            elif q.query_id not in qids:
                                queries.append(q)
                                qids.add(q.query_id)
                        del qids
                    curr_sess = (int(tokens[0]), int(tokens[2]), {})
                    curr_user = int(tokens[3])
                elif (
                    curr_sess is not None
                    and tlen > 4
                    and int(tokens[0]) == curr_sess[0]
                ):
                    t = int(tokens[1])
                    if last_click is not None:
                        query = curr_sess[2][last_click[0]]
                        query.click(last_click[1], t - last_click[2])
                        last_click = None
                    if tokens[2] == "Q":
                        serp_id = int(tokens[3])
                        query_id = int(tokens[4])
                        query_terms = tuple([int(s) for s in tokens[5].split(",")])
                        results = []
                        for r in tokens[6:]:
                            rs = r.split(",")
                            results.append((int(rs[0]), int(rs[1])))
                        query = LoggedQuery(curr_user, query_id, query_terms, results)
                        curr_sess[2][serp_id] = query
                    elif tokens[2] == "C":
                        last_click = (int(tokens[3]), int(tokens[4]), t)
                    else:
                        logging.warning(f"unknown record type: {tokens[2]}")
        logging.info(f"  loading time: {time.process_time() - st}")
        st = time.process_time()
        try:
            with open(pickle_file, "wb") as f:
                pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.error(f"{pickle_file} cannot be accessed.")
        logging.info(f"  saving time: {time.process_time() - st}")


def load_logged_queries(params) -> Sequence[TrainingQuery]:
    logging.info("loading logged queries...")
    if "folder" not in params:
        raise Exception('Please define "folder" in "raw_data"')
    folder = params["folder"] if "folder" in params else ""
    if len(folder) == 0:
        folder = os.getcwd()
    cache_file_name = params["cache_file_name"] if "cache_file_name" in params else ""
    cache_file = os.path.join(folder, f"{cache_file_name}.pickle")
    if len(cache_file_name) > 0 and os.access(cache_file, os.R_OK):
        logging.info(f"  loading {cache_file}")
        try:
            st = time.perf_counter()
            with open(cache_file, "rb") as f:
                logged_queries = pickle.load(f)
            logging.info(f"  loading time {time.perf_counter() - st}")
            return logged_queries
        except Exception as err:
            logging.warning(f" loading error {err}")
    base_file_name = params["base_file_name"] if "base_file_name" in params else ""
    if len(base_file_name) == 0:
        raise Exception('"base_file_name" not defined!')
    days = params["days"] if "days" in params else []
    all_queries = {}
    st = time.perf_counter()
    for day in days:
        pickle_file = os.path.join(folder, f"{base_file_name}_{day:02}.pickle")
        if os.access(pickle_file, os.R_OK):
            logging.info(f"  loading {pickle_file}")
            with open(pickle_file, "rb") as f:
                queries = pickle.load(f)
            if queries is None:
                logging.warning(f"  loading {pickle_file} failed!")
            else:
                logging.info(f"  loaded queries: {len(queries)}")
                for q in queries:
                    if q.query_id in all_queries:
                        tq = all_queries[q.query_id]
                    else:
                        tq = TrainingQuery(q.query_id, q.query_terms)
                        all_queries[q.query_id] = tq
                    tq.add(q)
        else:
            logging.warning(f"  {pickle_file} not accessible!")
    logging.info(f"  loading time {time.perf_counter() - st}")
    logged_queries = tuple(all_queries.values())
    for v in logged_queries:
        v.finalize()
    if len(cache_file_name) > 0:
        logging.info(f"  saving logged queries to {cache_file}")
        try:
            st = time.perf_counter()
            with open(cache_file, "wb") as f:
                pickle.dump(logged_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"  saving time {time.perf_counter() - st}")
        except Exception:
            logging.warning(f"  {cache_file} not accessible!")
    return logged_queries


class TrainingDataset:
    def __init__(self, params, device=None):
        if "folder" not in params:
            raise Exception('Please define "folder" in "dataset"')
        self._folder = params["folder"]
        self._days = params["days"] if "days" in params else []
        self._base_file_name = (
            params["base_file_name"] if "base_file_name" in params else "train"
        )
        self._min_query_count = (
            params["min_query_count"]
            if "min_query_count" in params
            else MIN_QUERY_COUNT
        )
        self._cache_file = params["cache_file"] if "cache_file" in params else ""

        self._device = device
        self._queries = None

        self._query_ids = None
        self._query_terms = None
        self._position_relevances = None

    def load_queries(self, reload=False):
        logging.info("loading training queries...")
        pickle_file = os.path.join(self._folder, self._cache_file)
        if not reload and len(self._cache_file) > 0 and os.access(pickle_file, os.R_OK):
            logging.info(f"  loading {pickle_file}")
            st = time.process_time()
            with open(pickle_file, "rb") as f:
                (
                    min_query_count,
                    days,
                    queries,
                    query_ids,
                    query_terms,
                    position_relevances,
                ) = pickle.load(f)
            if min_query_count != self._min_query_count or days != self._days:
                logging.info("  updated config from last cache, reload")
                self.load_queries(True)
            else:
                self._queries = queries
                self._query_ids = query_ids
                self._query_terms = query_terms
                self._position_relevances = position_relevances
                logging.info(
                    f"  loaded {len(self._queries)}, "
                    f"  time {time.process_time() - st}"
                )
        else:
            all_queries = {}
            for d in self._days:
                cache_file = os.path.join(
                    self._folder, f"{self._base_file_name}_{d:02}.pickle"
                )
                if os.access(cache_file, os.R_OK):
                    logging.info(f"  loading {cache_file}")
                    st = time.process_time()
                    with open(cache_file, "rb") as f:
                        queries = pickle.load(f)
                    if queries is None:
                        logging.warning(f"  loading {cache_file} failed!")
                        continue
                    logging.info(f"    loaded queries: {len(queries)}")
                    logging.info(f"    loading time {time.process_time() - st}")
                    st = time.process_time()
                    for q in queries:
                        if q.query_id not in all_queries:
                            qr = TrainingQuery(q.query_id, q.query_terms)
                            all_queries[q.query_id] = qr
                        else:
                            qr = all_queries[q.query_id]
                        qr.add(q)
                    logging.info(f"    process time {time.process_time() - st}")
                else:
                    logging.warning(f"    {cache_file} not accessible!")
            self._queries = []
            for v in all_queries.values():
                if v.count >= self._min_query_count:
                    v.finalize()
                    v.pack()
                    self._queries.append(v)
            self._query_ids = None
            self._query_terms = None
            self._position_relevances = None
            if len(self._cache_file) > 0:
                logging.info(f"saving training queries to {pickle_file}")
                try:
                    st = time.process_time()
                    with open(pickle_file, "wb") as f:
                        self._process_training_queries()
                        pickle.dump(
                            (
                                self._min_query_count,
                                self._days,
                                self._queries,
                                self._query_ids,
                                self._query_terms,
                                self._position_relevances,
                            ),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    logging.info(f"  saving time {time.process_time() - st}")
                except Exception:
                    logging.warning(f"  {pickle_file} not accessible!")
        # self._query_ids = None
        # self._query_terms = None
        # self._position_relevances = None
        logging.info(f"loaded training queries: {len(self._queries)}")

    def _process_training_queries(self):
        if (
            self._query_ids is not None
            and self._query_terms is not None
            and self._position_relevances is not None
        ):
            return
        logging.info("processing training queries...")
        st = time.process_time()
        self._query_ids = {}
        self._query_terms = {}
        self._position_relevances = [RunningAverage() for _ in range(MAX_SLATE_SIZE)]
        for q in self._queries:
            self._query_ids[q.query_id] = q
            for t in q.query_terms:
                if t in self._query_terms:
                    self._query_terms[t].merge(q)
                else:
                    mq = TrainingQuery(0, (t,))
                    mq.merge(q)
                    self._query_terms[t] = mq
            for ra, r in zip(self._position_relevances, q.position_relevances):
                ra.add(r)
        for q in self._query_terms.values():
            q.finalize()
        self._position_relevances = [v.average for v in self._position_relevances]
        logging.info(f"processing time {time.process_time() - st}")

    @property
    def training_queries(self):
        return self._queries

    def item_relevances(
        self, query_id: int, query_terms: Tuple[int], items: Iterable[Tuple[int, int]]
    ) -> SlateItemValues:
        self._process_training_queries()
        if query_id in self._query_ids:
            q = self._query_ids[query_id]
            rels = q.url_relevances
        else:
            ras = {}
            for t in query_terms:
                if t in self._query_terms:
                    q = self._query_terms[t]
                    for i, r in q.url_relevances:
                        if i in ras:
                            ra = ras[i]
                        else:
                            ra = RunningAverage()
                            ras[i] = ra
                        ra.add(r)
            rels = {i: r.average for i, r in ras.items()}
        item_rels = {}
        for i in items:
            if i in rels:
                item_rels[i] = rels[i]
            else:
                item_rels[i] = 0.0
        return SlateItemValues(item_rels)

    def slot_relevances(self, slots: SlateSlots) -> SlateSlotValues:
        return SlateSlotValues(self._position_relevances[: len(slots)])


class YandexSlateModel(SlateModel):
    def __init__(self, dataset: TrainingDataset):
        self._dataset = dataset

    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        query = context.query.value
        return self._dataset.item_relevances(query[0], query[1:])

    def slot_probabilities(self, context: SlateContext) -> SlateSlotValues:
        return self._dataset.slot_relevances(context.slots)


def evaluate(
    experiments: Iterable[Tuple[Iterable[SlateEstimator], int]],
    log_dataset: TrainingDataset,
    log_distribution: RewardDistribution,
    tgt_dataset: TrainingDataset,
    tgt_distribution: RewardDistribution,
    log_queries: Sequence[TrainingQuery],
    slate_size: int,
    item_size: int,
    metric_func: str,
    max_num_workers: int,
    device=None,
):
    log_length = len(log_queries)
    slots = SlateSlots(slate_size)

    logging.info("Generating log...")
    st = time.perf_counter()
    tasks = []
    total_samples = 0
    for estimators, num_samples in experiments:
        samples = []
        if num_samples * 10 > log_length:
            logging.warning(f"not enough log data, needs {num_samples * 10}")
            continue
        query_choices = np.random.choice(log_length, num_samples, replace=False)
        for i in query_choices:
            q = log_queries[i]
            context = SlateContext(SlateQuery((q.query_id, *(q.query_terms))), slots)
            url_relevances = q.url_relevances
            if len(url_relevances) > item_size:
                url_relevances = {
                    k: v
                    for k, v in sorted(
                        url_relevances.items(), key=lambda item: item[1]
                    )[:item_size]
                }
            items = url_relevances.keys()
            log_item_rewards = log_dataset.item_relevances(
                q.query_id, q.query_terms, items
            )
            log_item_probs = log_distribution(log_item_rewards)
            tgt_item_rewards = tgt_dataset.item_relevances(
                q.query_id, q.query_terms, items
            )
            tgt_item_probs = tgt_distribution(tgt_item_rewards)
            tgt_slot_expectation = tgt_item_probs.slot_item_expectations(slots)
            gt_item_rewards = SlateItemValues(url_relevances)
            if metric_func == "dcg":
                metric = DCGSlateMetric(device=device)
            elif metric_func == "err":
                metric = ERRSlateMetric(4.0, device=device)
            else:
                metric = NDCGSlateMetric(gt_item_rewards, device=device)
            slot_weights = metric.slot_weights(slots)
            if tgt_item_probs.is_deterministic:
                tgt_slate_prob = 1.0
                log_slate = tgt_item_probs.sample_slate(slots)
            else:
                tgt_slate_prob = float("nan")
                log_slate = log_item_probs.sample_slate(slots)
            log_slate_prob = log_item_probs.slate_probability(log_slate)
            log_rewards = log_slate.slot_values(gt_item_rewards)
            log_reward = metric.calculate_reward(slots, log_rewards, None, slot_weights)
            gt_slot_rewards = tgt_slot_expectation.expected_rewards(gt_item_rewards)
            gt_reward = metric.calculate_reward(
                slots, gt_slot_rewards, None, slot_weights
            )
            samples.append(
                LogSample(
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
                )
            )
            total_samples += 1
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

    # uncomment to create cache for faster data loading
    # create_cache(params["raw_data"])

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = None

    logging.info('loading "log_data"')
    log_dataset = TrainingDataset(params["log_data"])
    st = time.perf_counter()
    log_dataset.load_queries()
    logging.info(f"load time: {time.perf_counter() - st}")

    logging.info('loading "target_data"')
    tgt_dataset = TrainingDataset(params["target_data"])
    st = time.perf_counter()
    tgt_dataset.load_queries()
    logging.info(f"load time: {time.perf_counter() - st}")

    logging.info('loading "test_data"')
    st = time.perf_counter()
    log_queries = load_logged_queries(params["test_data"])
    logging.info(f"load time: {time.perf_counter() - st}")

    estimators = [IPSEstimator(), PseudoInverseEstimator(), PBMEstimator()]

    evaluate(
        [(estimators, 200)] * 4,
        log_dataset,
        RankingDistribution(1.0),
        tgt_dataset,
        FrechetDistribution(2.0, True),
        log_queries,
        5,
        10,
        "ndcg",
        2,
    )
