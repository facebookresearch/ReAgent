#!/usr/bin/env python3

import abc
import logging
import pickle
from typing import NamedTuple, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


try:
    # pyre-fixme[21]: Could not find `petastorm`.
    from petastorm import make_batch_reader

    # pyre-fixme[21]: Could not find module `petastorm.pytorch`.
    # pyre-fixme[21]: Could not find module `petastorm.pytorch`.
    from petastorm.pytorch import DataLoader, decimal_friendly_collate
except ModuleNotFoundError:
    logger.warn("petastorm is not installed; please install if you want to use this")


from reagent.core.parameters import NormalizationData
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
)
from reagent.workflow.types import (
    Dataset,
    ReaderOptions,
    RewardOptions,
    TableSpec,
)

from .reagent_data_module import ReAgentDataModule


class TrainEvalSampleRanges(NamedTuple):
    train_sample_range: Tuple[float, float]
    eval_sample_range: Tuple[float, float]


def get_sample_range(
    input_table_spec: TableSpec, calc_cpe_in_training: bool
) -> TrainEvalSampleRanges:
    table_sample = input_table_spec.table_sample
    eval_table_sample = input_table_spec.eval_table_sample

    if not calc_cpe_in_training:
        # use all data if table sample = None
        if table_sample is None:
            train_sample_range = (0.0, 100.0)
        else:
            train_sample_range = (0.0, table_sample)
        return TrainEvalSampleRanges(
            train_sample_range=train_sample_range,
            # eval samples will not be used
            eval_sample_range=(0.0, 0.0),
        )

    error_msg = (
        "calc_cpe_in_training is set to True. "
        f"Please specify table_sample(current={table_sample}) and "
        f"eval_table_sample(current={eval_table_sample}) such that "
        "eval_table_sample + table_sample <= 100. "
        "In order to reliably calculate CPE, eval_table_sample "
        "should not be too small."
    )
    assert table_sample is not None, error_msg
    assert eval_table_sample is not None, error_msg
    assert (eval_table_sample + table_sample) <= (100.0 + 1e-3), error_msg

    return TrainEvalSampleRanges(
        train_sample_range=(0.0, table_sample),
        eval_sample_range=(100.0 - eval_table_sample, 100.0),
    )


# pyre-fixme[13]: Attribute `_normalization_data_map` is never initialized.
# pyre-fixme[13]: Attribute `_train_dataset` is never initialized.
# pyre-fixme[13]: Attribute `_eval_dataset` is never initialized.
class ManualDataModule(ReAgentDataModule):
    _normalization_data_map: Dict[str, NormalizationData]
    _train_dataset: Dataset
    _eval_dataset: Optional[Dataset]

    def __init__(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        model_manager=None,
    ):
        super().__init__()
        self.input_table_spec = input_table_spec
        self.reward_options = reward_options or RewardOptions()
        self.reader_options = reader_options or ReaderOptions()
        self._model_manager = model_manager
        self.setup_data = setup_data
        self.saved_setup_data = saved_setup_data or {}

        self._setup_done = False

    def prepare_data(self, *args, **kwargs):
        if self.setup_data is not None:
            return None

        key = "normalization_data_map"

        normalization_data_map = (
            self.run_feature_identification(self.input_table_spec)
            if key not in self.saved_setup_data
            else pickle.loads(self.saved_setup_data[key])
        )
        calc_cpe_in_training = self.should_generate_eval_dataset
        sample_range_output = get_sample_range(
            self.input_table_spec, calc_cpe_in_training
        )
        train_dataset = self.query_data(
            input_table_spec=self.input_table_spec,
            sample_range=sample_range_output.train_sample_range,
            reward_options=self.reward_options,
        )
        eval_dataset = None
        if calc_cpe_in_training:
            eval_dataset = self.query_data(
                input_table_spec=self.input_table_spec,
                sample_range=sample_range_output.eval_sample_range,
                reward_options=self.reward_options,
            )

        return self._pickle_setup_data(
            normalization_data_map=normalization_data_map,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def _pickle_setup_data(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> Dict[str, bytes]:
        setup_data = dict(
            normalization_data_map=pickle.dumps(normalization_data_map),
            train_dataset=pickle.dumps(train_dataset),
            eval_dataset=pickle.dumps(eval_dataset),
        )
        self.setup_data = setup_data
        return setup_data

    def setup(self, stage=None):
        if self._setup_done:
            return

        setup_data = {k: pickle.loads(v) for k, v in self.setup_data.items()}

        self._normalization_data_map = setup_data["normalization_data_map"]
        self._train_dataset = setup_data["train_dataset"]
        self._eval_dataset = setup_data["eval_dataset"]

        self._setup_done = True

    @property
    def model_manager(self):
        model_manager = self._model_manager
        assert model_manager
        return model_manager

    @model_manager.setter
    def model_manager(self, model_manager):
        assert self._model_manager is None
        self._model_manager = model_manager

    def get_normalization_data_map(
        self, keys: List[str]
    ) -> Dict[str, NormalizationData]:
        return self._normalization_data_map

    @abc.abstractmethod
    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        """
        Derive preprocessing parameters from data. The keys of the dict should
        match the keys from `required_normalization_keys()`
        """
        pass

    @property
    @abc.abstractmethod
    def required_normalization_keys(self) -> List[str]:
        """ Get the normalization keys required for current instance """
        pass

    def __getattr__(self, attr):
        """ Get X_normalization_data by attribute """
        normalization_data_suffix = "_normalization_data"
        if attr.endswith(normalization_data_suffix):
            assert self._normalization_data_map is not None, (
                f"Trying to access {attr} but normalization_data_map "
                "has not been set via `initialize_trainer`."
            )
            normalization_key = attr[: -len(normalization_data_suffix)]
            normalization_data = self._normalization_data_map.get(
                normalization_key, None
            )
            if normalization_data is None:
                raise AttributeError(
                    f"normalization key `{normalization_key}` is unavailable. "
                    f"Available keys are: {self._normalization_data_map.keys()}."
                )
            return normalization_data

        raise AttributeError(
            f"attr {attr} not available {type(self)} (subclass of ModelManager)."
        )

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        """
        Massage input table into the format expected by the trainer
        """
        pass

    @abc.abstractmethod
    def build_batch_preprocessor(self) -> BatchPreprocessor:
        pass

    def get_dataloader(self, dataset: Dataset):
        batch_preprocessor = self.build_batch_preprocessor()
        reader_options = self.reader_options
        assert reader_options
        data_reader = make_batch_reader(
            # pyre-fixme[16]: `HiveDataSetClass` has no attribute `parquet_url`.
            dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        # NOTE: must be wrapped by DataLoaderWrapper to call __exit__() on end of epoch
        dataloader = DataLoader(
            data_reader,
            batch_size=reader_options.minibatch_size,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=batch_preprocessor, use_gpu=False
            ),
        )
        return _closing_iter(dataloader)

    def train_dataloader(self):
        return self.get_dataloader(self._train_dataset)

    def test_dataloader(self):
        # TODO: we currently use the same data for test and validation.
        # We should have three different splits of the total data
        return self._get_eval_dataset()

    def val_dataloader(self):
        return self._get_eval_dataset()

    def _get_eval_dataset(self):
        test_dataset = getattr(self, "_eval_dataset", None)
        if not test_dataset:
            return None
        return self.get_dataloader(test_dataset)


def _closing_iter(dataloader):
    yield from dataloader
    dataloader.__exit__(None, None, None)


def collate_and_preprocess(batch_preprocessor: BatchPreprocessor, use_gpu: bool):
    """Helper for Petastorm's DataLoader to preprocess.
    TODO(kaiwenw): parallelize preprocessing by using transform of Petastorm reader
    Should pin memory and preprocess in reader and convert to gpu in collate_fn.
    """

    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch

    return collate_fn
