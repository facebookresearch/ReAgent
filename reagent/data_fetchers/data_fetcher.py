#!/usr/bin/env python3


import logging
from typing import Dict, Optional

from reagent.core.types import Dataset, PreprocessingOptions, ReaderOptions, TableSpec
from reagent.parameters import NormalizationParameters
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor


logger = logging.getLogger(__name__)


class DataFetcher:
    # TODO: T71636145 Make a more specific API for DataFetcher
    def query_data(self, **kwargs):
        raise NotImplementedError()

    # TODO: T71636145 Make a more specific API for DataFetcher
    def query_data_parametric(self, **kwargs):
        raise NotImplementedError()

    def identify_normalization_parameters(
        self,
        table_spec: TableSpec,
        column_name: str,
        preprocessing_options: PreprocessingOptions,
        seed: Optional[int] = None,
    ) -> Dict[int, NormalizationParameters]:
        raise NotImplementedError()

    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_preprocessor: Optional[BatchPreprocessor],
        use_gpu: bool,
        reader_options: ReaderOptions,
    ):
        raise NotImplementedError()
