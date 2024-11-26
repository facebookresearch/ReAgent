#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import dataclasses
import logging
import time
from typing import Dict, Optional

import torch

# pyre-fixme[21]: Could not find module `reagent.core.parameters`.
from reagent.core.parameters import NormalizationData

# pyre-fixme[21]: Could not find module `reagent.core.tensorboardX`.
from reagent.core.tensorboardX import summary_writer_context

# pyre-fixme[21]: Could not find module `reagent.data.manual_data_module`.
from reagent.data.manual_data_module import get_sample_range

# pyre-fixme[21]: Could not find module `reagent.data.oss_data_fetcher`.
from reagent.data.oss_data_fetcher import OssDataFetcher

# pyre-fixme[21]: Could not find module `reagent.model_managers.model_manager`.
from reagent.model_managers.model_manager import ModelManager

# pyre-fixme[21]: Could not find module `reagent.model_managers.union`.
from reagent.model_managers.union import ModelManager__Union

# pyre-fixme[21]: Could not find module `reagent.publishers.union`.
from reagent.publishers.union import ModelPublisher__Union

# pyre-fixme[21]: Could not find module `reagent.validators.union`.
from reagent.validators.union import ModelValidator__Union

# pyre-fixme[21]: Could not find module `reagent.workflow.env`.
from reagent.workflow.env import get_new_named_entity_ids, get_workflow_id

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import (
    Dataset,
    ModuleNameToEntityId,
    ReaderOptions,
    RecurringPeriod,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)

# pyre-fixme[21]: Could not find module `torch.utils.tensorboard`.
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


def identify_and_train_network(
    # pyre-fixme[11]: Annotation `TableSpec` is not defined as a type.
    input_table_spec: TableSpec,
    # pyre-fixme[11]: Annotation `ModelManager__Union` is not defined as a type.
    model: ModelManager__Union,
    num_epochs: int,
    use_gpu: Optional[bool] = None,
    # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
    reward_options: Optional[RewardOptions] = None,
    # pyre-fixme[11]: Annotation `ReaderOptions` is not defined as a type.
    reader_options: Optional[ReaderOptions] = None,
    # pyre-fixme[11]: Annotation `ResourceOptions` is not defined as a type.
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None,
    # pyre-fixme[11]: Annotation `ModelValidator__Union` is not defined as a type.
    validator: Optional[ModelValidator__Union] = None,
    # pyre-fixme[11]: Annotation `ModelPublisher__Union` is not defined as a type.
    publisher: Optional[ModelPublisher__Union] = None,
    # pyre-fixme[11]: Annotation `RLTrainingOutput` is not defined as a type.
) -> RLTrainingOutput:
    if use_gpu is None:
        # pyre-fixme[35]: Target cannot be annotated.
        # pyre-fixme[16]: Module `torch` has no attribute `cuda`.
        use_gpu: bool = torch.cuda.is_available()

    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    reward_options = reward_options or RewardOptions()
    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    reader_options = reader_options or ReaderOptions()

    manager = model.value

    normalization_data_map = None
    setup_data = None

    data_module = manager.get_data_module(
        input_table_spec=input_table_spec,
        reward_options=reward_options,
        reader_options=reader_options,
        resource_options=resource_options,
    )
    if data_module is not None:
        data_module.prepare_data()
        setup_data = data_module.setup_data
    else:
        normalization_data_map = manager.run_feature_identification(input_table_spec)

    return query_and_train(
        input_table_spec,
        model,
        num_epochs,
        use_gpu=use_gpu,
        setup_data=setup_data,
        normalization_data_map=normalization_data_map,
        reward_options=reward_options,
        reader_options=reader_options,
        resource_options=resource_options,
        warmstart_path=warmstart_path,
        validator=validator,
        publisher=publisher,
    )


def query_and_train(
    input_table_spec: TableSpec,
    model: ModelManager__Union,
    num_epochs: int,
    use_gpu: bool,
    *,
    setup_data: Optional[Dict[str, bytes]] = None,
    saved_setup_data: Optional[Dict[str, bytes]] = None,
    # pyre-fixme[11]: Annotation `NormalizationData` is not defined as a type.
    normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    reward_options: Optional[RewardOptions] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidator__Union] = None,
    publisher: Optional[ModelPublisher__Union] = None,
    # pyre-fixme[11]: Annotation `ModuleNameToEntityId` is not defined as a type.
    named_model_ids: Optional[ModuleNameToEntityId] = None,
    # pyre-fixme[11]: Annotation `RecurringPeriod` is not defined as a type.
    recurring_period: Optional[RecurringPeriod] = None,
) -> RLTrainingOutput:
    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    child_workflow_id = get_workflow_id()
    if named_model_ids is None:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        named_model_ids = get_new_named_entity_ids(model.value.serving_module_names())

    logger.info("Starting query")

    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    reward_options = reward_options or RewardOptions()
    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    reader_options = reader_options or ReaderOptions()
    # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
    resource_options = resource_options or ResourceOptions()
    manager = model.value

    resource_options.gpu = int(use_gpu)

    if saved_setup_data is not None:

        def _maybe_get_bytes(v) -> bytes:
            if isinstance(v, bytes):
                return v

            # HACK: FBLearner sometimes pack bytes into Blob
            return v.data

        saved_setup_data = {k: _maybe_get_bytes(v) for k, v in saved_setup_data.items()}

    if setup_data is None:
        data_module = manager.get_data_module(
            input_table_spec=input_table_spec,
            reward_options=reward_options,
            reader_options=reader_options,
            resource_options=resource_options,
            saved_setup_data=saved_setup_data,
        )
        if data_module is not None:
            data_module.prepare_data()
            setup_data = data_module.setup_data
            # Throw away existing normalization data map
            normalization_data_map = None

    if sum([int(setup_data is not None), int(normalization_data_map is not None)]) != 1:
        raise ValueError("setup_data and normalization_data_map are mutually exclusive")

    train_dataset = None
    eval_dataset = None
    # pyre-fixme[16]: Module `reagent` has no attribute `data`.
    data_fetcher = OssDataFetcher()
    if normalization_data_map is not None:
        calc_cpe_in_training = manager.should_generate_eval_dataset
        # pyre-fixme[16]: Module `reagent` has no attribute `data`.
        sample_range_output = get_sample_range(input_table_spec, calc_cpe_in_training)
        train_dataset = manager.query_data(
            input_table_spec=input_table_spec,
            sample_range=sample_range_output.train_sample_range,
            reward_options=reward_options,
            data_fetcher=data_fetcher,
        )
        eval_dataset = None
        if calc_cpe_in_training:
            eval_dataset = manager.query_data(
                input_table_spec=input_table_spec,
                sample_range=sample_range_output.eval_sample_range,
                reward_options=reward_options,
                data_fetcher=data_fetcher,
            )

    logger.info("Starting training")

    results = train_workflow(
        manager,
        train_dataset,
        eval_dataset,
        num_epochs=num_epochs,
        use_gpu=use_gpu,
        setup_data=setup_data,
        normalization_data_map=normalization_data_map,
        named_model_ids=named_model_ids,
        child_workflow_id=child_workflow_id,
        reward_options=reward_options,
        reader_options=reader_options,
        resource_options=resource_options,
        warmstart_path=warmstart_path,
    )

    if validator is not None:
        results = run_validator(validator, results)

    if publisher is not None:
        results = run_publisher(
            publisher,
            model,
            results,
            setup_data,
            named_model_ids,
            child_workflow_id,
            recurring_period,
        )

    return results


def train_workflow(
    # pyre-fixme[11]: Annotation `ModelManager` is not defined as a type.
    model_manager: ModelManager,
    # pyre-fixme[11]: Annotation `Dataset` is not defined as a type.
    train_dataset: Optional[Dataset],
    eval_dataset: Optional[Dataset],
    *,
    num_epochs: int,
    use_gpu: bool,
    named_model_ids: ModuleNameToEntityId,
    child_workflow_id: int,
    setup_data: Optional[Dict[str, bytes]] = None,
    normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    reward_options: Optional[RewardOptions] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None,
) -> RLTrainingOutput:
    # pyre-fixme[16]: Module `torch` has no attribute `utils`.
    writer = SummaryWriter()
    logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

    if setup_data is not None:
        data_module = model_manager.get_data_module(
            setup_data=setup_data,
            reward_options=reward_options,
            reader_options=reader_options,
            resource_options=resource_options,
        )
        assert data_module is not None
        data_module.setup()
    else:
        data_module = None

    if normalization_data_map is None:
        assert data_module is not None
        normalization_data_map = data_module.get_normalization_data_map()

    warmstart_input_path = warmstart_path or None
    trainer_module = model_manager.build_trainer(
        use_gpu=use_gpu,
        reward_options=reward_options,
        normalization_data_map=normalization_data_map,
    )

    if not reader_options:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        reader_options = ReaderOptions()

    if not resource_options:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        resource_options = ResourceOptions()

    # pyre-fixme[16]: Module `reagent` has no attribute `core`.
    with summary_writer_context(writer):
        train_output, lightning_trainer = model_manager.train(
            trainer_module,
            train_dataset,
            eval_dataset,
            None,
            data_module,
            num_epochs,
            reader_options,
            resource_options,
            checkpoint_path=warmstart_input_path,
        )

    output_paths = {}
    for module_name, serving_module in model_manager.build_serving_modules(
        trainer_module, normalization_data_map
    ).items():
        torchscript_output_path = f"{model_manager.__class__.__name__}_{module_name}_{round(time.time())}.torchscript"
        # pyre-fixme[16]: Module `torch` has no attribute `jit`.
        torch.jit.save(serving_module, torchscript_output_path)
        logger.info(f"Saved {module_name} to {torchscript_output_path}")
        output_paths[module_name] = torchscript_output_path
    return dataclasses.replace(train_output, output_paths=output_paths)


def run_validator(
    validator: ModelValidator__Union, training_output: RLTrainingOutput
) -> RLTrainingOutput:
    assert (
        training_output.validation_result is None
    ), f"validation_output was set to f{training_output.validation_output}"
    model_validator = validator.value
    validation_result = model_validator.validate(training_output)
    return dataclasses.replace(training_output, validation_result=validation_result)


def run_publisher(
    publisher: ModelPublisher__Union,
    model_chooser: ModelManager__Union,
    training_output: RLTrainingOutput,
    setup_data: Optional[Dict[str, bytes]],
    recurring_workflow_ids: ModuleNameToEntityId,
    child_workflow_id: int,
    recurring_period: Optional[RecurringPeriod],
) -> RLTrainingOutput:
    assert (
        training_output.publishing_result is None
    ), f"publishing_output was set to f{training_output.publishing_output}"
    model_publisher = publisher.value
    model_manager = model_chooser.value
    publishing_result = model_publisher.publish(
        model_manager,
        training_output,
        setup_data,
        recurring_workflow_ids,
        child_workflow_id,
        recurring_period,
    )
    return dataclasses.replace(training_output, publishing_result=publishing_result)
