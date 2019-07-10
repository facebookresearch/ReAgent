#!/usr/bin/env python3

from typing import Tuple, Union, cast

import torch
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.types import (
    DiscreteDqnInput,
    FeatureVector,
    ParametricDqnInput,
    PolicyNetworkInput,
    SARSAInput,
    TrainingBatch,
)


class BatchPreprocessor:
    def __call__(self, batch: TrainingBatch) -> TrainingBatch:
        raise NotImplementedError()


class DiscreteDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(self, state_preprocessor: Preprocessor):
        self.state_preprocessor = state_preprocessor

    def __call__(self, batch: TrainingBatch) -> TrainingBatch:
        training_input = cast(
            Union[DiscreteDqnInput, ParametricDqnInput], batch.training_input
        )

        preprocessed_state = self.state_preprocessor(
            training_input.state.float_features.value,
            training_input.state.float_features.presence,
        )
        preprocessed_next_state = self.state_preprocessor(
            training_input.next_state.float_features.value,
            training_input.next_state.float_features.presence,
        )
        new_training_input = training_input._replace(
            state=training_input.state._replace(float_features=preprocessed_state),
            next_state=training_input.next_state._replace(
                float_features=preprocessed_next_state
            ),
        )
        return batch._replace(training_input=new_training_input)


class ParametricDqnBatchPreprocessor(DiscreteDqnBatchPreprocessor):
    def __init__(
        self, state_preprocessor: Preprocessor, action_preprocessor: Preprocessor
    ):
        super().__init__(state_preprocessor)
        self.action_preprocessor = action_preprocessor

    def __call__(self, batch: TrainingBatch) -> TrainingBatch:
        batch = super().__call__(batch)

        if isinstance(batch.training_input, ParametricDqnInput):
            training_input = cast(ParametricDqnInput, batch.training_input)
            preprocessed_tiled_next_state = self.state_preprocessor(
                training_input.tiled_next_state.float_features.value,
                training_input.tiled_next_state.float_features.presence,
            )
            preprocessed_action = self.action_preprocessor(
                training_input.action.float_features.value,
                training_input.action.float_features.presence,
            )
            preprocessed_next_action = self.action_preprocessor(
                training_input.next_action.float_features.value,
                training_input.next_action.float_features.presence,
            )
            preprocessed_possible_actions = self.action_preprocessor(
                training_input.possible_actions.float_features.value,
                training_input.possible_actions.float_features.presence,
            )
            preprocessed_possible_next_actions = self.action_preprocessor(
                training_input.possible_next_actions.float_features.value,
                training_input.possible_next_actions.float_features.presence,
            )
            return batch._replace(
                training_input=training_input._replace(
                    action=training_input.action._replace(
                        float_features=preprocessed_action
                    ),
                    next_action=training_input.next_action._replace(
                        float_features=preprocessed_next_action
                    ),
                    possible_actions=training_input.possible_actions._replace(
                        float_features=preprocessed_possible_actions
                    ),
                    possible_next_actions=training_input.possible_next_actions._replace(
                        float_features=preprocessed_possible_next_actions
                    ),
                    tiled_next_state=training_input.tiled_next_state._replace(
                        float_features=preprocessed_tiled_next_state
                    ),
                )
            )
        elif isinstance(batch.training_input, SARSAInput):
            training_input_sarsa = cast(SARSAInput, batch.training_input)
            preprocessed_tiled_next_state = self.state_preprocessor(
                training_input_sarsa.tiled_next_state.float_features.value,  # type: ignore
                training_input_sarsa.tiled_next_state.float_features.presence,  # type: ignore
            )
            preprocessed_action = self.action_preprocessor(
                training_input_sarsa.action.float_features.value,  # type: ignore
                training_input_sarsa.action.float_features.presence,  # type: ignore
            )
            preprocessed_next_action = self.action_preprocessor(
                training_input_sarsa.next_action.float_features.value,  # type: ignore
                training_input_sarsa.next_action.float_features.presence,  # type: ignore
            )
            return batch._replace(
                training_input=training_input_sarsa._replace(
                    action=training_input_sarsa.action._replace(  # type: ignore
                        float_features=preprocessed_action
                    ),
                    next_action=training_input_sarsa.next_action._replace(  # type: ignore
                        float_features=preprocessed_next_action
                    ),
                    tiled_next_state=training_input_sarsa.tiled_next_state._replace(  # type: ignore
                        float_features=preprocessed_tiled_next_state
                    ),
                )
            )
        else:
            assert False, "Invalid training_input type: " + str(
                type(batch.training_input)
            )


class PolicyNetworkBatchPreprocessor(DiscreteDqnBatchPreprocessor):
    def __init__(
        self, state_preprocessor: Preprocessor, action_preprocessor: Preprocessor
    ):
        super().__init__(state_preprocessor)
        self.action_preprocessor = action_preprocessor

    def __call__(self, batch: TrainingBatch) -> TrainingBatch:
        batch = super().__call__(batch)

        training_input = cast(PolicyNetworkInput, batch.training_input)

        action_before_preprocessing = cast(FeatureVector, training_input.action)
        preprocessed_action = self.action_preprocessor(
            action_before_preprocessing.float_features.value,
            action_before_preprocessing.float_features.presence,
        )
        next_action_before_preprocessing = cast(
            FeatureVector, training_input.next_action
        )
        preprocessed_next_action = self.action_preprocessor(
            next_action_before_preprocessing.float_features.value,
            next_action_before_preprocessing.float_features.presence,
        )
        return batch._replace(
            training_input=training_input._replace(
                action=action_before_preprocessing._replace(
                    float_features=preprocessed_action
                ),
                next_action=next_action_before_preprocessing._replace(
                    float_features=preprocessed_next_action
                ),
            )
        )
