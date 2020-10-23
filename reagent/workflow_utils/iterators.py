#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import OrderedDict

from reagent.core.tracker import observable
from reagent.tensorboardX import SummaryWriterContext
from torch.utils.data import IterableDataset

# pyre-fixme[21]: Could not find module `tqdm`.
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@observable(epoch_start=int, epoch_end=int)
class EpochIterator:
    def __init__(self, num_epochs: int):
        assert num_epochs > 0
        self.num_epochs = num_epochs

    def __iter__(self):
        SummaryWriterContext._reset_globals()
        for epoch in range(self.num_epochs):
            self.notify_observers(epoch_start=epoch)
            yield epoch
            self.notify_observers(epoch_end=epoch)
            # TODO: flush at end of epoch?


def get_batch_size(batch):
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


class DataLoaderWrapper(IterableDataset):
    def __init__(self, dataloader: IterableDataset, dataloader_size: int):
        """Wraps around an Iterable Dataloader to report progress bars and
        increase global step of SummaryWriter. At last iteration, will call
        dataloader.__exit__ if needed (e.g. Petastorm DataLoader).

        Args:
            dataloader: the iteratable dataloader to wrap around
            dataloader_size: size of the dataset we're iterating over
        """

        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.dataloader_size = dataloader_size

    def __iter__(self):
        t = tqdm(total=self.dataloader_size, desc="iterating dataloader")
        for batch in self.dataloader:
            batch_size = get_batch_size(batch)
            yield batch
            t.update(batch_size)
            SummaryWriterContext.increase_global_step()

        # clean up if need to (e.g. Petastorm Dataloader)
        if hasattr(self.dataloader, "__exit__"):
            self.dataloader.__exit__(None, None, None)
