#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import queue
import random
import sys
import threading
from typing import NamedTuple

import torch
import torch.multiprocessing as multiprocessing
import torch.utils.data._utils
from torch._six import string_classes
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils.signal_handling import (
    _remove_worker_pids,
    _set_SIGCHLD_handler,
    _set_worker_pids,
    _set_worker_signal_handlers,
)
from torch.utils.data._utils.worker import ManagerWatchdog


MANAGER_STATUS_CHECK_INTERVAL = 5.0


WorkerDone = collections.namedtuple("WorkerDone", ["worker_id"])


def _worker_loop(
    data_reader,
    batch_queue,
    data_queue,
    global_done_event,
    worker_done_event,
    seed,
    init_fn,
    worker_id,
):
    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    # TODO: numpy doesn't take seed bigger than INT32
    # np.random.seed(seed)
    torch.manual_seed(seed)

    # Do not wait for putting thread to join when this worker exits. Otherwise,
    # this worker may always be waiting to put and doesn't check batch_queue
    # and global_done_event for termination signal.
    data_queue.cancel_join_thread()

    if init_fn is not None:
        init_fn(worker_id)

    watchdog = ManagerWatchdog()

    shard = data_reader.get_shard(worker_id)
    shard_itr = iter(shard)

    shard_done = False

    while True:
        if shard_done:
            # Wait until the main thread acknowledge the WorkerDone message or
            # it signals shutdown.
            if (
                not watchdog.is_alive()
                or global_done_event.is_set()
                or worker_done_event.wait(0.1)
            ):
                break
            continue

        try:
            idx = batch_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            if watchdog.is_alive() and not global_done_event.is_set():
                continue
            else:
                break
        # use global_done_event so that we can get faster exiting signal even if there
        # are still batches in batch_queue
        if idx is None or global_done_event.is_set():
            break
        try:
            samples = next(shard_itr)
        except StopIteration:
            # Signal to the main thread that this worker has run out of data.
            # The worker cannot exit immediately because the queue might not be
            # flushed immediately.
            data_queue.put((idx, WorkerDone(worker_id)))
            shard_done = True
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
            del samples


def _pin_memory_loop(in_queue, out_queue, done_event, pin_memory, device_id):
    """
    This is copied from dataloader. It uses a different `pin_memory_batch()`.
    It'd probably be best to merge.
    """
    if pin_memory:
        torch.cuda.set_device(device_id)

    while True:
        try:
            r = in_queue.get()
        except Exception:
            if done_event.is_set():
                return
            raise
        if r is None or done_event.is_set():
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            if pin_memory:
                batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


def pin_memory_batch(batch):
    """
    This is ripped off from dataloader. The only difference is that it preserves
    the type of Mapping so that the OrderedDict is maintained.
    """
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory().cuda(non_blocking=True)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, NamedTuple) or hasattr(batch, "_asdict"):
        return type(batch)(
            **{name: pin_memory_batch(value) for name, value in batch._asdict().items()}
        )
    elif isinstance(batch, collections.Mapping):
        # NB: preserving OrderedDict
        return type(batch)((k, pin_memory_batch(sample)) for k, sample in batch.items())
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


class _DataStreamerIter(object):
    r"""Iterates once over the DataStreamer's data_reader"""

    def __init__(self, streamer):
        self.data_reader = streamer.data_reader
        self.num_workers = streamer.num_workers
        self.pin_memory = streamer.pin_memory and torch.cuda.is_available()
        self.timeout = streamer.timeout

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = streamer.worker_init_fn
            self.worker_result_queue = multiprocessing.Queue()
            self.batch_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.done_event = multiprocessing.Event()

            self.worker_done_events = [
                multiprocessing.Event() for _i in range(self.num_workers)
            ]
            self.workers = []
            for i in range(self.num_workers):
                w = multiprocessing.Process(
                    target=_worker_loop,
                    args=(
                        self.data_reader,
                        self.batch_queue,
                        self.worker_result_queue,
                        self.done_event,
                        self.worker_done_events[i],
                        base_seed + i,
                        self.worker_init_fn,
                        i,
                    ),
                )
                w.daemon = True  # ensure that the worker exits on process exit
                # Process.start() actually take some time as it needs to start a
                # process and pass the arguments over via a pipe. Therefore, we
                # only add a worker to self.workers list after it started, so
                # that we do not call .join() if program dies before it starts,
                # and __del__ tries to join it but will get:
                #     AssertionError: can only join a started process.
                w.start()
                self.workers.append(w)

            self.num_live_workers = self.num_workers

            if self.pin_memory:
                self.data_queue = queue.Queue()
                self.pin_memory_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        self.done_event,
                        self.pin_memory,
                        torch.cuda.current_device(),
                    ),
                )
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            _set_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()
        else:
            # No workers
            self.data_reader_iter = iter(self.data_reader)

    def _get_batch(self):
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError(
                    "DataReader timed out after {} seconds".format(self.timeout)
                )
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            batch = next(self.data_reader_iter)  # May raise StopIteration
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert not self.shutdown and self.batches_outstanding > 0
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            self._put_indices()
            if isinstance(batch, WorkerDone):
                # Acknowledge receiving so worker can terminate early
                self.worker_done_events[batch.worker_id].set()
                self.num_live_workers -= 1
                if self.num_live_workers == 0:
                    self._shutdown_workers()
                    raise StopIteration
                else:
                    continue

            if isinstance(batch, ExceptionWrapper):
                raise batch.exc_type(batch.exc_msg)
            return batch

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        self.batch_queue.put(self.send_idx)
        self.batches_outstanding += 1
        self.send_idx += 1

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataReaderIter cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            # removes pids from the C side data structure first so worker
            # termination afterwards won't trigger false positive error report.
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False
            self.done_event.set()
            if self.pin_memory:
                # Sending `None` to `pin_memory_thread` must be before
                # stopping worker processes because the workers may leave
                # corrupted data in `worker_result_queue`, causing
                # `pin_memory_thread` unable to read and terminate properly.
                self.worker_result_queue.put(None)
            # Workers can't be waiting to put be cause their output queue
            # is a multiprocessing.Queue and its .put is non-blocking.
            # They can only be waiting to get, so we put `None` here.
            for _w in self.workers:
                # Putting as many None as workers to ensure worker will get one
                self.batch_queue.put(None)
            for w in self.workers:
                w.join()
            if self.pin_memory:
                self.pin_memory_thread.join()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataStreamer(object):
    r"""
    Data streamer. Provides single- or multi-process iterators over the data_reader.

    Arguments:
        data_reader (DataReader): data_reader from which to stream the data.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        pin_memory (bool, optional): If ``True``, the data streamer will copy tensors
            into CUDA pinned memory before returning them.
        timeout (numeric, optional): if positive, the timeout value for collecting a
            batch from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`datastreamer-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    def __init__(
        self,
        data_reader,
        num_workers=0,
        pin_memory=False,
        timeout=0,
        worker_init_fn=None,
    ):
        self.data_reader = data_reader
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if self.num_workers < 0:
            raise ValueError(
                "num_workers option cannot be negative; "
                "use num_workers=0 to disable multiprocessing."
            )

    def __iter__(self):
        return _DataStreamerIter(self)
