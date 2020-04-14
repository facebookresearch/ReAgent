#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from typing import List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from reagent.ope.estimators.slate_estimators import (
    DMEstimator,
    LogEpisode,
    LogSample,
    NDCGSlateMetric,
    SlateContext,
    SlateEstimatorInput,
    SlateItemProbabilities,
    SlateItems,
    SlateItemValues,
    SlateModel,
    SlateQuery,
    SlateSlotItemExpectations,
    SlateSlots,
    SlateSlotValues,
    make_slate,
)
from reagent.ope.utils import RunningAverage


# Slate test using Yandex Personalized Web Search Dataset:
#   https://www.kaggle.com/c/yandex-personalized-web-search-challenge/

RELEVANT_THRESHOLD = 49
HIGHLY_RELEVANT_THRESHOLD = 399
MAX_POSITION = 10
MIN_QUERY_COUNT = 10


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
        self._position_relevances = [0.0] * max(len(self._list), MAX_POSITION)
        self._url_relevances = {}
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


class ProcessedQuery:
    def __init__(self, query_id: int, query_terms: Tuple[int]):
        self._query_id = query_id
        self._query_terms = query_terms
        self._count = 0
        self._url_relevances: Union[
            Sequence[Tuple[Tuple[int, int], float]],
            MutableMapping[Tuple[int, int], float],
        ] = {}
        self._position_relevances = [0.0] * MAX_POSITION

    def add(self, query: LoggedQuery):
        if len(query.clicks) == 0:
            return
        self._count += 1
        urs = query.url_relevances
        for item_id, r in urs.items():
            if item_id not in self._url_relevances:
                self._url_relevances[item_id] = 0.0
            else:
                self._url_relevances[item_id] += r
        prs = query.position_relevances
        for i in range(MAX_POSITION):
            self._position_relevances[i] += prs[i]

    def merge(self, other: "ProcessedQuery"):
        self._count += 1
        for i, r in other.url_relevances.items():
            if i not in self._url_relevances:
                self._url_relevances[i] = r
            else:
                self._url_relevances[i] += r
        for i in range(MAX_POSITION):
            self._position_relevances[i] += other.position_relevances[i]

    def finalize(self):
        self._url_relevances = {
            k: v / self._count for k, v in self._url_relevances.items()
        }
        self._position_relevances = [v / self._count for v in self._position_relevances]

    def pack(self):
        self._url_relevances = list(self._url_relevances.items())

    def unpack(self):
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


def load_logged_queries(params):
    logging.info("loading logged queries...")
    if "folder" not in params:
        raise Exception('Please define "folder" in "raw_data"')
    folder = params["folder"] if "folder" in params else ""
    if len(folder) == 0:
        folder = os.getcwd()
    cache_folder = params["cache_folder"] if "cache_folder" in params else folder
    if len(cache_folder) == 0:
        cache_folder = folder
    base_file_name = params["base_file_name"] if "base_file_name" in params else ""
    if len(base_file_name) == 0:
        raise Exception('"base_file_name" not defined!')
    days = params["days"] if "days" in params else []
    all_queries = {}
    st = time.process_time()
    for day in days:
        cache_file = f"{base_file_name}_{day:02}.pickle"
        pickle_file = os.path.join(cache_folder, cache_file)
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
                        all_queries[q.q.query_id].append(q)
                    else:
                        all_queries[q.q.query_id] = [q]
        else:
            logging.warning(f"  {pickle_file} not accessible!")
    logging.info(f"loading time {time.process_time() - st}")
    return all_queries


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
                min_query_count, days, queries = pickle.load(f)
            if min_query_count != self._min_query_count or days != self._days:
                logging.info("  updated config from last cache, reload")
                self.load_queries(True)
            else:
                self._queries = queries
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
                            qr = ProcessedQuery(q.query_id, q.query_terms)
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
            if len(self._cache_file) > 0:
                logging.info(f"saving training queries to {pickle_file}")
                try:
                    st = time.process_time()
                    with open(pickle_file, "wb") as f:
                        pickle.dump(
                            (self._min_query_count, self._days, self._queries),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    logging.info(f"  saving time {time.process_time() - st}")
                except Exception:
                    logging.warning(f"  {pickle_file} not accessible!")
        self._query_ids = None
        self._query_terms = None
        self._position_relevances = None
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
        self._position_relevances = [RunningAverage() for _ in range(MAX_POSITION)]
        for q in self._queries:
            q.unpack()
            self._query_ids[q.query_id] = q
            for t in q.query_terms:
                if t in self._query_terms:
                    self._query_terms[t].merge(q)
                else:
                    mq = ProcessedQuery(0, (t,))
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

    def predict_item(self, query_id: int, query_terms: Tuple[int]) -> SlateItemValues:
        self._process_training_queries()
        if query_id in self._query_ids:
            q = self._query_ids[query_id]
            return SlateItemValues(dict(q.url_relevances.items()))
        else:
            rels = {}
            for t in query_terms:
                q = self._query_terms[t]
                for i, r in q.url_relevances:
                    if i in rels:
                        ra = rels[i]
                    else:
                        ra = RunningAverage()
                    ra.add(r)
            return SlateItemValues({i: r.average for i, r in rels.items()})

    def predict_slot(self, slots: SlateSlots) -> SlateSlotItemExpectations:
        return SlateSlotItemExpectations(self._position_relevances[: len(slots)])


class YandexSlateModel(SlateModel):
    def __init__(self, dataset: TrainingDataset):
        self._dataset = dataset

    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        query = context.query.value
        return self._dataset.predict_item(query[0], query[1:])

    def slot_probabilities(self, context: SlateContext) -> SlateSlotItemExpectations:
        return self._dataset.predict_slot(context.slots)


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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.info('loading "ground_truth_training_data"')
    ground_truth_training_dataset = TrainingDataset(
        params["ground_truth_training_data"]
    )
    st = time.process_time()
    ground_truth_training_dataset.load_queries()
    logging.info(f"load time: {time.process_time() - st}")
    gt_model = YandexSlateModel(ground_truth_training_dataset)

    logging.info('loading "log_training_data"')
    log_training_dataset = TrainingDataset(params["log_training_data"])
    st = time.process_time()
    log_training_dataset.load_queries()
    logging.info(f"load time: {time.process_time() - st}")

    logging.info('loading "target_training_data"')
    tgt_training_dataset = TrainingDataset(params["target_training_data"])
    st = time.process_time()
    tgt_training_dataset.load_queries()
    logging.info(f"load time: {time.process_time() - st}")
    tgt_model = YandexSlateModel(tgt_training_dataset)

    log_queries = load_logged_queries(params["test_data"])
    slots = SlateSlots(MAX_POSITION)
    episodes = []
    for qid, qs in sorted(log_queries.items(), key=lambda i: len(i[1]), reverse=True):
        log_query = qs[0]
        context = SlateContext(SlateQuery((qid, *(log_query.query_terms))), slots)
        log_item_rewards = log_training_dataset.predict_item(
            log_query.query_id, log_query.query_terms
        )
        log_item_probs = SlateItemProbabilities(log_item_rewards.values)
        tgt_item_rewards = tgt_model.item_rewards(context)
        tgt_item_probs = SlateItemProbabilities(tgt_item_rewards.values)
        gt_item_rewards = gt_model.item_rewards(context)
        metric = NDCGSlateMetric(gt_item_rewards)
        samples = []
        for q in qs:
            slate = make_slate(slots, q.list)
            samples.append(
                LogSample(
                    slate,
                    slate.slot_values(gt_item_rewards),
                    SlateSlotValues(q.position_relevances),
                )
            )
        episodes.append(
            LogEpisode(
                context,
                metric,
                samples,
                None,
                log_item_probs,
                None,
                tgt_item_probs,
                gt_item_rewards,
            )
        )
    input = SlateEstimatorInput(episodes)

    estimator = DMEstimator()
    logging.info("Evaluating...")
    st = time.process_time()
    rs = estimator.evaluate(input)
    dt = time.process_time() - st
    logging.info(f"Evaluating DMEstimator done: {rs} in {dt}s")
