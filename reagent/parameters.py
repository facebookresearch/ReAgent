#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List, Optional

from reagent.core.configuration import param_hash
from reagent.core.dataclasses import dataclass, field
from reagent.parameters_seq2slate import LearningMethod, RewardClamp
from reagent.types import BaseDataClass


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
    softmax_policy: int = 1
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
class RainbowDQNParameters(BaseDataClass):
    double_q_learning: bool = True
    dueling_architecture: bool = True
    # Batch constrained q-learning (bcq) is not technically a Rainbow addition
    # but an augmentation to DQN so putting here.
    bcq: bool = False
    # 0 = max q-learning, 1 = imitation learning
    bcq_drop_threshold: float = 0.1
    categorical: bool = False
    num_atoms: int = 51
    qmin: float = -100
    qmax: float = 200
    # C51's performance degrades with l2_regularization != 0.
    c51_l2_decay: float = 0
    quantile: bool = False


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
class CNNParameters(BaseDataClass):
    __hash__ = param_hash

    conv_dims: List[int]
    conv_height_kernels: List[int]
    conv_width_kernels: List[int]
    pool_kernels_strides: List[int]
    pool_types: List[str]
    num_input_channels: int
    input_height: int
    input_width: int


@dataclass(frozen=True)
class FeedForwardParameters(BaseDataClass):
    __hash__ = param_hash

    layers: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_layer_norm: Optional[bool] = None


@dataclass(frozen=True)
class TrainingParameters(BaseDataClass):
    __hash__ = param_hash

    minibatch_size: int = 4096
    learning_rate: float = 0.001
    optimizer: str = "ADAM"
    layers: List[int] = field(default_factory=lambda: [-1, 256, 128, 1])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu", "linear"])
    lr_policy: str = "fixed"
    lr_decay: float = 0.999
    dropout_ratio: float = 0.0
    warm_start_model_path: Optional[str] = None
    cnn_parameters: Optional[CNNParameters] = None
    l2_decay: float = 0.01
    weight_init_min_std: float = 0.0
    use_batch_norm: bool = False
    clip_grad_norm: Optional[float] = None
    minibatches_per_step: int = 1
    do_not_warm_start_optimizer: Optional[bool] = None


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
class DiscreteActionModelParameters(BaseDataClass):
    __hash__ = param_hash

    actions: List[str] = field(default_factory=lambda: [])
    rl: RLParameters = RLParameters()
    training: TrainingParameters = TrainingParameters()
    rainbow: RainbowDQNParameters = RainbowDQNParameters()
    state_feature_params: Optional[StateFeatureParameters] = None
    target_action_distribution: Optional[List[float]] = None
    evaluation: EvaluationParameters = EvaluationParameters()


@dataclass(frozen=True)
class ContinuousActionModelParameters(BaseDataClass):
    rl: RLParameters
    training: TrainingParameters
    rainbow: RainbowDQNParameters
    state_feature_params: Optional[StateFeatureParameters] = None
    evaluation: EvaluationParameters = EvaluationParameters()


@dataclass(frozen=True)
class OptimizerParameters(BaseDataClass):
    optimizer: str = "ADAM"
    learning_rate: float = 0.001
    l2_decay: float = 0.01


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
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class BaselineParameters(BaseDataClass):
    dim_feedforward: int
    num_stacked_layers: int
    warmup_num_batches: int = 0
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class Seq2SlateTransformerParameters(BaseDataClass):
    transformer: TransformerParameters
    baseline: Optional[BaselineParameters]
    on_policy: bool
    learning_method: LearningMethod
    importance_sampling_clamp_max: Optional[float] = None
    simulation_reward_clamp: Optional[RewardClamp] = None
    # penalize sequences far away from prod
    simulation_distance_penalty: Optional[float] = None


@dataclass(frozen=True)
class RankingParameters(BaseDataClass):
    minibatch_size: int
    max_src_seq_len: int
    max_tgt_seq_len: int
    greedy_serving: bool
