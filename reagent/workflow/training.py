#!/usr/bin/env python3

import dataclasses
import logging
from typing import Dict, NamedTuple, Optional, Tuple

from reagent.parameters import NormalizationData
from reagent.workflow.env import get_workflow_id
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.publishers.union import ModelPublisher__Union
from reagent.workflow.types import (
    RecurringPeriod,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from reagent.workflow.validators.union import ModelValidator__Union


logger = logging.getLogger(__name__)


def identify_and_train_network(
    input_table_spec: TableSpec,
    model: ModelManager__Union,
    num_epochs: int,
    use_gpu: bool = True,
    reward_options: Optional[RewardOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidator__Union] = None,
    publisher: Optional[ModelPublisher__Union] = None,
) -> RLTrainingOutput:
    manager = model.value
    normalization_data_map = manager.run_feature_identification(input_table_spec)

    return query_and_train(
        input_table_spec,
        model,
        normalization_data_map,
        num_epochs,
        use_gpu=use_gpu,
        reward_options=reward_options,
        warmstart_path=warmstart_path,
        validator=validator,
        publisher=publisher,
    )


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


def query_and_train(
    input_table_spec: TableSpec,
    model: ModelManager__Union,
    normalization_data_map: Dict[str, NormalizationData],
    num_epochs: int,
    use_gpu: bool,
    reward_options: Optional[RewardOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidator__Union] = None,
    publisher: Optional[ModelPublisher__Union] = None,
    parent_workflow_id: Optional[int] = None,
    recurring_period: Optional[RecurringPeriod] = None,
) -> RLTrainingOutput:
    child_workflow_id = get_workflow_id()
    if parent_workflow_id is None:
        parent_workflow_id = child_workflow_id

    logger.info("Starting query")

    reward_options = reward_options or RewardOptions()
    manager = model.value

    calc_cpe_in_training = manager.should_generate_eval_dataset
    sample_range_output = get_sample_range(input_table_spec, calc_cpe_in_training)
    train_dataset = manager.query_data(
        input_table_spec=input_table_spec,
        sample_range=sample_range_output.train_sample_range,
        reward_options=reward_options,
    )
    eval_dataset = None
    if calc_cpe_in_training:
        eval_dataset = manager.query_data(
            input_table_spec=input_table_spec,
            sample_range=sample_range_output.eval_sample_range,
            reward_options=reward_options,
        )

    logger.info("Starting training")
    results = manager.train_workflow(
        train_dataset,
        eval_dataset,
        normalization_data_map,
        num_epochs,
        use_gpu,
        parent_workflow_id=parent_workflow_id,
        child_workflow_id=child_workflow_id,
        reward_options=reward_options,
        warmstart_path=warmstart_path,
    )

    if validator is not None:
        results = run_validator(validator, results)

    if publisher is not None:
        results = run_publisher(
            publisher,
            model,
            results,
            parent_workflow_id,
            child_workflow_id,
            recurring_period,
        )

    return results


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
    recurring_workflow_id: int,
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
        recurring_workflow_id,
        child_workflow_id,
        recurring_period,
    )
    return dataclasses.replace(training_output, publishing_result=publishing_result)
