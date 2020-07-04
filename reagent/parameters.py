#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List, Optional

from reagent.base_dataclass import BaseDataClass
from reagent.core.configuration import param_hash
from reagent.core.dataclasses import dataclass, field
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters_seq2slate import LearningMethod, RewardClamp


# For TD3 and SAC: actions are normalized in this range for training and
# rescaled back to action_space.low/high at serving time.
CONTINUOUS_TRAINING_ACTION_RANGE = (-1.0, 1.0)


@dataclass(frozen=True)
class RLParameters(BaseDataClass):
    __hash__ = param_hash

    gamma: float = 0.9
    epsilon: float = 0.1
    target_update_rate: float = 0.001
    maxq_learning: bool = True
    reward_boost: Optional[Dict[str, float]] = None
    temperature: float = 0.01
    softmax_policy: bool = True
    use_seq_num_diff_as_time_diff: bool = False
    q_network_loss: str = "mse"
    set_missing_value_to_zero: bool = False
    tensorboard_logging_freq: int = 0
    predictor_atol_check: float = 0.0
    predictor_rtol_check: float = 5e-5
    time_diff_unit_length: float = 1.0
    multi_steps: Optional[int] = None
    # for pytorch discrete model, specify the max number of prediction change
    # allowed during conversions between model frameworks in ratio
    ratio_different_predictions_tolerance: float = 0


@dataclass(frozen=True)
class MDNRNNTrainerParameters(BaseDataClass):
    __hash__ = param_hash

    hidden_size: int = 64
    num_hidden_layers: int = 2
    minibatch_size: int = 16
    learning_rate: float = 0.001
    num_gaussians: int = 5
    train_data_percentage: float = 60.0
    validation_data_percentage: float = 20.0
    test_data_percentage: float = 20.0
    # weight in calculating world-model loss
    reward_loss_weight: float = 1.0
    next_state_loss_weight: float = 1.0
    not_terminal_loss_weight: float = 1.0
    fit_only_one_next_step: bool = False


@dataclass(frozen=True)
class Seq2RewardTrainerParameters(BaseDataClass):
    __hash__ = param_hash

    minibatch_size: int = 16
    learning_rate: float = 0.001
    train_data_percentage: float = 60.0
    validation_data_percentage: float = 20.0
    test_data_percentage: float = 20.0
    multi_steps: int = 1
    action_names: List[str] = field(default_factory=lambda: [])
    batch_size: int = 32
    calc_cpe_in_training: bool = True


@dataclass(frozen=True)
class CEMTrainerParameters(BaseDataClass):
    __hash__ = param_hash

    plan_horizon_length: int = 0
    num_world_models: int = 0
    cem_population_size: int = 0
    cem_num_iterations: int = 0
    ensemble_population_size: int = 0
    num_elites: int = 0
    mdnrnn: MDNRNNTrainerParameters = MDNRNNTrainerParameters()
    rl: RLParameters = RLParameters()
    alpha: float = 0.25
    epsilon: float = 0.001


@dataclass(frozen=True)
class EvaluationParameters(BaseDataClass):
    calc_cpe_in_training: bool = True


@dataclass(frozen=True)
class EvolutionParameters(BaseDataClass):
    population_size: int = 1000
    mutation_power: float = 0.1
    learning_rate: float = 0.01


@dataclass(frozen=True)
class StateFeatureParameters(BaseDataClass):
    __hash__ = param_hash

    state_feature_names_override: List[str] = field(default_factory=lambda: [])
    state_feature_hashes_override: List[int] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class NormalizationParameters(BaseDataClass):
    __hash__ = param_hash

    feature_type: str
    boxcox_lambda: Optional[float] = None
    boxcox_shift: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    possible_values: Optional[List[int]] = None  # Assume present for ENUM type
    quantiles: Optional[
        List[float]
    ] = None  # Assume present for QUANTILE type and sorted
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class NormalizationKey(object):
    """ Keys for dictionaries of NormalizationData """

    STATE = "state"
    ACTION = "action"
    ITEM = "item"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class NormalizationData(BaseDataClass):
    __hash__ = param_hash
    dense_normalization_parameters: Dict[int, NormalizationParameters]


#################################################
#             RL Ranking parameters             #
#################################################
@dataclass(frozen=True)
class TransformerParameters(BaseDataClass):
    num_heads: int
    dim_model: int
    dim_feedforward: int
    num_stacked_layers: int


@dataclass(frozen=True)
class GRUParameters(BaseDataClass):
    dim_model: int
    num_stacked_layers: int


@dataclass(frozen=True)
class BaselineParameters(BaseDataClass):
    dim_feedforward: int
    num_stacked_layers: int
    warmup_num_batches: int = 0


@dataclass(frozen=True)
class Seq2SlateParameters(BaseDataClass):
    on_policy: bool = True
    learning_method: LearningMethod = LearningMethod.REINFORCEMENT_LEARNING
    importance_sampling_clamp_max: Optional[float] = None
    simulation_reward_clamp: Optional[RewardClamp] = None
    # penalize sequences far away from prod
    simulation_distance_penalty: Optional[float] = None


@dataclass(frozen=True)
class RankingParameters(BaseDataClass):
    max_src_seq_len: int
    max_tgt_seq_len: int
    greedy_serving: bool
