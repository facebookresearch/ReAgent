#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import ml.rl.types as rlt
import torch
from ml.rl.models.base import ModelBase
from ml.rl.preprocessing.postprocessor import Postprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor
from torch import nn


logger = logging.getLogger(__name__)


# TODO: The feature definition should be ModelFeatureConfig


class DiscreteDqnWithPreprocessor(ModelBase):
    """
    This is separate from DiscreteDqnPredictorWrapper so that we can pass typed inputs
    into the model. This is possible because JIT only traces tensor operation.
    In contrast, JIT scripting needs to compile the code, therefore, it won't recognize
    any custom Python type.
    """

    def __init__(self, model: ModelBase, state_preprocessor: Preprocessor):
        super().__init__()
        self.model = model
        self.state_preprocessor = state_preprocessor

    def forward(self, state_with_presence: Tuple[torch.Tensor, torch.Tensor]):
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )
        state_feature_vector = rlt.PreprocessedState.from_tensor(preprocessed_state)
        q_values = self.model(state_feature_vector).q_values
        return q_values

    def input_prototype(self):
        return (self.state_preprocessor.input_prototype(),)

    @property
    def sorted_features(self):
        # TODO: the interface here should be ModelFeatureConfig
        return self.state_preprocessor.sorted_features


class DiscreteDqnPredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t"]

    def __init__(
        self,
        dqn_with_preprocessor: DiscreteDqnWithPreprocessor,
        action_names: List[str],
    ) -> None:
        super().__init__()

        self.state_sorted_features_t = dqn_with_preprocessor.sorted_features

        self.dqn_with_preprocessor = torch.jit.trace(
            dqn_with_preprocessor, dqn_with_preprocessor.input_prototype()
        )
        self.action_names = torch.jit.Attribute(action_names, List[str])

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        """
        This interface is used by DiscreteDqnTorchPredictor
        """
        return self.state_sorted_features_t

    @torch.jit.script_method
    def forward(
        self, state_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[List[str], torch.Tensor]:
        q_values = self.dqn_with_preprocessor(state_with_presence)
        return (self.action_names, q_values)


class ParametricDqnWithPreprocessor(ModelBase):
    def __init__(
        self,
        model: ModelBase,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
    ):
        super().__init__()
        self.model = model
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor

    @property
    def state_sorted_features(self) -> List[int]:
        return self.state_preprocessor.sorted_features

    @property
    def action_sorted_features(self) -> List[int]:
        return self.action_preprocessor.sorted_features

    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ):
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )
        preprocessed_action = self.action_preprocessor(
            action_with_presence[0], action_with_presence[1]
        )
        state_feature_vector = rlt.PreprocessedStateAction.from_tensors(
            state=preprocessed_state, action=preprocessed_action
        )
        q_value = self.model(state_feature_vector).q_value
        return q_value

    def input_prototype(self):
        return (
            self.state_preprocessor.input_prototype(),
            self.action_preprocessor.input_prototype(),
        )


class ParametricDqnPredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t", "action_sorted_features_t"]

    def __init__(self, dqn_with_preprocessor: ParametricDqnWithPreprocessor) -> None:
        super().__init__()

        self.state_sorted_features_t = dqn_with_preprocessor.state_sorted_features
        self.action_sorted_features_t = dqn_with_preprocessor.action_sorted_features
        self.dqn_with_preprocessor = torch.jit.trace(
            dqn_with_preprocessor, dqn_with_preprocessor.input_prototype()
        )

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        """
        This interface is used by ParametricDqnTorchPredictor
        """
        return self.state_sorted_features_t

    @torch.jit.script_method
    def action_sorted_features(self) -> List[int]:
        """
        This interface is used by ParametricDqnTorchPredictor
        """
        return self.action_sorted_features_t

    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[List[str], torch.Tensor]:
        value = self.dqn_with_preprocessor(state_with_presence, action_with_presence)
        return (["Q"], value)


class ActorWithPreprocessor(ModelBase):
    """
    This is separate from ActorPredictorWrapper so that we can pass typed inputs
    into the model. This is possible because JIT only traces tensor operation.
    In contrast, JIT scripting needs to compile the code, therefore, it won't recognize
    any custom Python type.
    """

    def __init__(
        self,
        model: ModelBase,
        state_preprocessor: Preprocessor,
        action_postprocessor: Optional[Postprocessor] = None,
    ):
        super().__init__()
        self.model = model
        self.state_preprocessor = state_preprocessor
        self.action_postprocessor = action_postprocessor

    def forward(self, state_with_presence: Tuple[torch.Tensor, torch.Tensor]):
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )
        state_feature_vector = rlt.PreprocessedState.from_tensor(preprocessed_state)
        # TODO: include log_prob in the output
        action = self.model(state_feature_vector).action
        if self.action_postprocessor:
            action = self.action_postprocessor(action)
        return action

    def input_prototype(self):
        return (self.state_preprocessor.input_prototype(),)

    @property
    def sorted_features(self):
        # TODO: the interface here should be ModelFeatureConfig
        return self.state_preprocessor.sorted_features


class ActorPredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t"]

    def __init__(self, actor_with_preprocessor: ActorWithPreprocessor) -> None:
        super().__init__()

        self.state_sorted_features_t = actor_with_preprocessor.sorted_features

        self.actor_with_preprocessor = torch.jit.trace(
            actor_with_preprocessor, actor_with_preprocessor.input_prototype()
        )

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        """
        This interface is used by ActorTorchPredictor
        """
        return self.state_sorted_features_t

    @torch.jit.script_method
    def forward(
        self, state_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        action = self.actor_with_preprocessor(state_with_presence)
        return action
