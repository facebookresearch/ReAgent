#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import ml.rl.types as rlt
import torch
from ml.rl.models.base import ModelBase
from ml.rl.models.seq2slate import RANK_MODE, Seq2SlateTransformerNet
from ml.rl.preprocessing.postprocessor import Postprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor
from torch import nn


logger = logging.getLogger(__name__)


# TODO: The feature definition should be ModelFeatureConfig


class DiscreteDqnWithPreprocessor(ModelBase):
    """
    This is separated from DiscreteDqnPredictorWrapper so that we can pass typed inputs
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


class Seq2SlateWithPreprocessor(ModelBase):
    def __init__(
        self,
        model: Seq2SlateTransformerNet,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        greedy: bool,
    ):
        super().__init__()
        self.model = model
        self.state_preprocessor = state_preprocessor
        self.candidate_preprocessor = candidate_preprocessor
        self.greedy = greedy

    def input_prototype(self):
        candidate_input_prototype = self.candidate_preprocessor.input_prototype()
        return (
            self.state_preprocessor.input_prototype(),
            (
                candidate_input_prototype[0].repeat((1, self.model.max_src_seq_len, 1)),
                candidate_input_prototype[1].repeat((1, self.model.max_src_seq_len, 1)),
            ),
        )

    @property
    def state_sorted_features(self) -> List[int]:
        return self.state_preprocessor.sorted_features

    @property
    def candidate_sorted_features(self) -> List[int]:
        return self.candidate_preprocessor.sorted_features

    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        candidate_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ):
        # state_value.shape == state_presence.shape == batch_size x state_feat_num
        # candidate_value.shape == candidate_presence.shape ==
        # batch_size x max_src_seq_len x candidate_feat_num
        batch_size = state_with_presence[0].shape[0]

        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )
        preprocessed_candidates = self.candidate_preprocessor(
            candidate_with_presence[0].view(
                batch_size * self.model.max_src_seq_len,
                len(self.candidate_sorted_features),
            ),
            candidate_with_presence[1].view(
                batch_size * self.model.max_src_seq_len,
                len(self.candidate_sorted_features),
            ),
        ).view(batch_size, self.model.max_src_seq_len, -1)

        # TODO: consider different numbers of candidates in the same batch_
        src_src_mask = torch.ones(
            batch_size, self.model.max_src_seq_len, self.model.max_src_seq_len
        )
        ranking_input = rlt.PreprocessedRankingInput.from_tensors(
            state=preprocessed_state,
            src_seq=preprocessed_candidates,
            src_src_mask=src_src_mask,
        )
        ranking_output = self.model(
            ranking_input,
            mode=RANK_MODE,
            tgt_seq_len=self.model.max_tgt_seq_len,
            greedy=self.greedy,
        )
        return ranking_output.ranked_tgt_out_probs, ranking_output.ranked_tgt_out_idx


class Seq2SlatePredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t", "candidate_sorted_features_t"]

    def __init__(self, seq2slate_with_preprocessor: Seq2SlateWithPreprocessor) -> None:
        super().__init__()

        self.state_sorted_features_t = seq2slate_with_preprocessor.state_sorted_features
        self.candidate_sorted_features_t = (
            seq2slate_with_preprocessor.candidate_sorted_features
        )
        self.seq2slate_with_preprocessor = torch.jit.trace(
            seq2slate_with_preprocessor, seq2slate_with_preprocessor.input_prototype()
        )

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        """
        This interface is used by Seq2SlateTorchPredictor
        """
        return self.state_sorted_features_t

    @torch.jit.script_method
    def candidate_sorted_features(self) -> List[int]:
        """
        This interface is used by Seq2SlateTorchPredictor
        """
        return self.candidate_sorted_features_t

    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        candidate_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ranked_tgt_out_probs shape: batch_size, tgt_seq_len, candidate_size
        # ranked_tgt_out_idx shape: batch_size, tgt_seq_len
        ranked_tgt_out_probs, ranked_tgt_out_idx = self.seq2slate_with_preprocessor(
            state_with_presence, candidate_with_presence
        )
        # ranked_tgt_out_probs shape: batch_size
        ranked_tgt_out_probs = torch.prod(
            torch.gather(
                ranked_tgt_out_probs, 2, ranked_tgt_out_idx.unsqueeze(-1)
            ).squeeze(),
            -1,
        )
        # -2 to offset padding symbol and decoder start symbol
        ranked_tgt_out_idx -= 2
        return ranked_tgt_out_probs, ranked_tgt_out_idx
