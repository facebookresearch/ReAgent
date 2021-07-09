#!/usr/bin/env python3

from typing import Optional

from reagent.core.registry_meta import wrap_oss_with_dataclass
from reagent.core.tagged_union import TaggedUnion

from .categorical_dqn.categorical import Categorical as CategoricalType
from .continuous_actor.dirichlet_fully_connected import (
    DirichletFullyConnected as DirichletFullyConnectedType,
)
from .continuous_actor.fully_connected import (
    FullyConnected as FullyConnectedContinuousActorType,
)
from .continuous_actor.gaussian_fully_connected import (
    GaussianFullyConnected as GaussianFullyConnectedType,
)
from .discrete_actor.fully_connected import (
    FullyConnected as FullyConnectedDiscreteActorType,
)
from .discrete_dqn.dueling import Dueling as DuelingType
from .discrete_dqn.fully_connected import FullyConnected as FullyConnectedType
from .discrete_dqn.fully_connected_with_embedding import (
    FullyConnectedWithEmbedding as FullyConnectedWithEmbeddingType,
)
from .parametric_dqn.fully_connected import (
    FullyConnected as FullyConnectedParametricType,
)
from .quantile_dqn.dueling_quantile import DuelingQuantile as DuelingQuantileType
from .quantile_dqn.quantile import Quantile as QuantileType
from .synthetic_reward.ngram_synthetic_reward import (
    NGramSyntheticReward as NGramSyntheticRewardType,
    NGramConvNetSyntheticReward as NGramConvNetSyntheticRewardType,
)
from .synthetic_reward.sequence_synthetic_reward import (
    SequenceSyntheticReward as SequenceSyntheticRewardType,
)
from .synthetic_reward.single_step_synthetic_reward import (
    SingleStepSyntheticReward as SingleStepSyntheticRewardType,
)
from .synthetic_reward.transformer_synthetic_reward import (
    TransformerSyntheticReward as TransformerSyntheticRewardType,
)
from .value.fully_connected import FullyConnected as FullyConnectedValueType
from .value.seq2reward_rnn import Seq2RewardNetBuilder as Seq2RewardNetBuilderType


@wrap_oss_with_dataclass
class DiscreteActorNetBuilder__Union(TaggedUnion):
    FullyConnected: Optional[FullyConnectedDiscreteActorType] = None


@wrap_oss_with_dataclass
class ContinuousActorNetBuilder__Union(TaggedUnion):
    FullyConnected: Optional[FullyConnectedContinuousActorType] = None
    DirichletFullyConnected: Optional[DirichletFullyConnectedType] = None
    GaussianFullyConnected: Optional[GaussianFullyConnectedType] = None


@wrap_oss_with_dataclass
class DiscreteDQNNetBuilder__Union(TaggedUnion):
    Dueling: Optional[DuelingType] = None
    FullyConnected: Optional[FullyConnectedType] = None
    FullyConnectedWithEmbedding: Optional[FullyConnectedWithEmbeddingType] = None


@wrap_oss_with_dataclass
class CategoricalDQNNetBuilder__Union(TaggedUnion):
    Categorical: Optional[CategoricalType] = None


@wrap_oss_with_dataclass
class QRDQNNetBuilder__Union(TaggedUnion):
    Quantile: Optional[QuantileType] = None
    DuelingQuantile: Optional[DuelingQuantileType] = None


@wrap_oss_with_dataclass
class ParametricDQNNetBuilder__Union(TaggedUnion):
    FullyConnected: Optional[FullyConnectedParametricType] = None


@wrap_oss_with_dataclass
class ValueNetBuilder__Union(TaggedUnion):
    FullyConnected: Optional[FullyConnectedValueType] = None
    Seq2RewardNetBuilder: Optional[Seq2RewardNetBuilderType] = None


@wrap_oss_with_dataclass
class SyntheticRewardNetBuilder__Union(TaggedUnion):
    SingleStepSyntheticReward: Optional[SingleStepSyntheticRewardType] = None
    NGramSyntheticReward: Optional[NGramSyntheticRewardType] = None
    NGramConvNetSyntheticReward: Optional[NGramConvNetSyntheticRewardType] = None
    SequenceSyntheticReward: Optional[SequenceSyntheticRewardType] = None
    TransformerSyntheticReward: Optional[TransformerSyntheticRewardType] = None
