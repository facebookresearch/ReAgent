# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

namespace py ml.rl.thrift.core

struct RLParameters {
  1: double gamma = 0.9,
  2: double epsilon = 0.1,
  3: double target_update_rate = 0.001,
  # 4: i32 reward_burnin = 1,
  5: bool maxq_learning = true,
  6: map<string, double> reward_boost,
  7: double temperature = 0.01,
  8: i32 softmax_policy = 1,
  9: bool use_seq_num_diff_as_time_diff = false,
  10: string q_network_loss = 'mse',
  11: bool set_missing_value_to_zero = false,
  12: optional i32 tensorboard_logging_freq,
  13: double predictor_atol_check = 0.0,
  14: double predictor_rtol_check = 5e-5,
  15: double time_diff_unit_length = 1.0,
  16: optional i32 multi_steps,
  # for pytorch discrete model, specify the max number of prediction change
  # allowed during conversions between model frameworks in ratio
  17: optional double ratio_different_predictions_tolerance,
}

struct RainbowDQNParameters {
  1: bool double_q_learning = true,
  2: bool dueling_architecture = true,
  # Batch constrained q-learning (bcq) is not technically a Rainbow addition,
  # but an augmentation to DQN so putting here.
  3: bool bcq = false,
  # 0 = max q-learning, 1 = imitation learning
  4: double bcq_drop_threshold = 0.1,
  5: bool categorical = false,
  6: i32 num_atoms = 51,
  7: double qmin = -100,
  8: double qmax = 200,
  # C51's performance degrades with l2_regularization != 0.
  9: double c51_l2_decay=0,
  10: bool quantile = false
}

struct CNNParameters {
  1: list<i32> conv_dims,
  2: list<i32> conv_height_kernels,
  3: list<i32> conv_width_kernels,
  4: list<i32> pool_kernels_strides,
  5: list<string> pool_types,
  6: i32 num_input_channels,
  7: i32 input_height,
  8: i32 input_width,
}

struct FeedForwardParameters {
  1: list<i32> layers = [256, 128],
  2: list<string> activations = ['relu', 'relu'],
  3: optional bool use_layer_norm,
}

struct FactorizationParameters {
  1: FeedForwardParameters state,
  2: FeedForwardParameters action,
}

struct TrainingParameters {
  1: i32 minibatch_size = 4096,
  2: double learning_rate = 0.001,
  3: string optimizer = 'ADAM',
  4: list<i32> layers = [-1, 256, 128, 1],
  5: list<string> activations = ['relu', 'relu', 'linear'],
  6: string lr_policy = 'fixed',
  7: double lr_decay = 0.999,
  8: double dropout_ratio = 0.0,
  9: optional string warm_start_model_path,
  10: optional CNNParameters cnn_parameters,
  11: optional FactorizationParameters factorization_parameters,
  12: double l2_decay = 0.01,
  14: double weight_init_min_std = 0.0,
  15: bool use_batch_norm = false,
  16: optional double clip_grad_norm,
  17: optional i32 minibatches_per_step,
}

struct EvaluationParameters {
  1: bool calc_cpe_in_training = true,
}

struct EvolutionParameters {
  1: i32 population_size = 1000,
  2: double mutation_power = 0.1,
  3: double learning_rate = 0.01,
}

struct StateFeatureParameters {
  1: list<string> state_feature_names_override = [],
  2: list<i32> state_feature_hashes_override = [],
}

struct DiscreteActionModelParameters {
  1: list<string> actions,
  2: RLParameters rl,
  3: TrainingParameters training,
  5: RainbowDQNParameters rainbow,
  7: optional StateFeatureParameters state_feature_params,
  8: optional list<double> target_action_distribution,
  # Some fields were removed; the next number to use is 11
  11: EvaluationParameters evaluation = {},
}

struct ContinuousActionModelParameters {
  1: RLParameters rl,
  2: TrainingParameters training,
  4: RainbowDQNParameters rainbow,
  # Some fields were removed; the next number to use is 6
  6: EvaluationParameters evaluation = {},
  7: optional StateFeatureParameters state_feature_params,
}

struct DDPGNetworkParameters {
  1: list<i32> layers = [-1, 256, 128, 1],
  2: list<string> activations = ['relu', 'relu', 'tanh'],
  3: double l2_decay = 0.01,
  4: double learning_rate = 0.001,
}

struct DDPGTrainingParameters {
  1: i32 minibatch_size = 2048,
  2: double final_layer_init = 0.003,
  3: string optimizer = 'ADAM',
  4: optional string warm_start_model_path,
  5: optional i32 minibatches_per_step,
}

struct DDPGModelParameters {
  1: RLParameters rl,
  2: DDPGTrainingParameters shared_training,
  3: DDPGNetworkParameters actor_training,
  4: DDPGNetworkParameters critic_training,
  5: optional map<i64, list<double>> action_rescale_map = {},
  6: optional StateFeatureParameters state_feature_params,
  7: EvaluationParameters evaluation = {},
}

struct OptimizerParameters {
  1: string optimizer = 'ADAM',
  2: double learning_rate = 0.001,
  3: double l2_decay = 0.01,
}

struct TD3TrainingParameters {
  1: i32 minibatch_size = 64,
  2: OptimizerParameters q_network_optimizer = {},
  3: OptimizerParameters actor_network_optimizer = {},
  4: bool use_2_q_functions = true,
  5: double exploration_noise = 0.2,
  6: i32 initial_exploration_ts = 1000,
  7: double target_policy_smoothing = 0.2,
  8: double noise_clip = 0.5,
  9: i32 delayed_policy_update = 2,
  10: optional string warm_start_model_path,
  11: optional i32 minibatches_per_step,
}

struct TD3ModelParameters {
  1: RLParameters rl = {
    "maxq_learning": false,
    "tensorboard_logging_freq": 100,
  },
  2: TD3TrainingParameters training = {},
  3: FeedForwardParameters q_network = {},
  4: FeedForwardParameters actor_network = {},
  5: optional StateFeatureParameters state_feature_params,
  6: EvaluationParameters evaluation = {},
}

struct SACTrainingParameters {
  1: i32 minibatch_size = 1024,
  2: OptimizerParameters q_network_optimizer = {},
  3: optional OptimizerParameters value_network_optimizer,
  4: OptimizerParameters actor_network_optimizer = {},
  5: bool use_2_q_functions = true,
  # alpha in the paper; controlling explore & exploit
  6: optional double entropy_temperature,
  7: optional string warm_start_model_path,
  8: bool logged_action_uniform_prior = true,
  9: optional i32 minibatches_per_step,
  10: bool use_value_network = true,
  11: optional double target_entropy,
  12: optional OptimizerParameters alpha_optimizer
  13: optional double action_embedding_kld_weight,
  14: optional list<double> action_embedding_mean,
  15: optional list<double> action_embedding_variance,
}

struct SACModelParameters {
  1: RLParameters rl = {
    "maxq_learning": false,
    "tensorboard_logging_freq": 100,
  },
  2: SACTrainingParameters training = {},
  3: FeedForwardParameters q_network = {},
  4: optional FeedForwardParameters value_network,
  5: FeedForwardParameters actor_network = {},
  8: optional StateFeatureParameters state_feature_params,
  9: EvaluationParameters evaluation = {},
  # constrain action output to sum to 1 (using dirichlet distribution)
  10: bool constrain_action_sum = false,
  11: optional bool do_not_preprocess_action,
}

struct KNNDQNModelParameters {
  1: RLParameters rl,
  2: DDPGTrainingParameters shared_training,
  3: DDPGNetworkParameters actor_training,
  4: DDPGNetworkParameters critic_training,
  5: i64 num_actions,
  6: i32 action_dim,
  7: i64 k,
  8: EvaluationParameters evaluation = {},
}

struct MDNRNNParameters {
  1: i32 hidden_size = 64,
  2: i32 num_hidden_layers = 2,
  3: i32 minibatch_size = 16,
  4: double learning_rate = 0.001,
  5: i32 num_gaussians = 5,
  6: double train_data_percentage = 60.0,
  7: double validation_data_percentage = 20.0,
  8: double test_data_percentage = 20.0,
  # weight in calculating world-model loss
  9: double reward_loss_weight = 1.0,
  10: double next_state_loss_weight = 1.0,
  11: double not_terminal_loss_weight = 1.0,
  12: bool fit_only_one_next_step = false,
}

struct CEMParameters {
  1: MDNRNNParameters mdnrnn,
  2: RLParameters rl,
  3: i32 plan_horizon_length,
  4: i32 num_world_models,
  5: i32 cem_population_size,
  6: i32 cem_num_iterations,
  7: i32 ensemble_population_size,
  8: i32 num_elites,
  9: double alpha = 0.25,
  10: double epsilon = 0.001,
  11: EvaluationParameters evaluation = {},
}
