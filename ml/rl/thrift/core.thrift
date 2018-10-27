namespace py ml.rl.thrift.core

struct AdditionalFeatureTypes {
  1: bool int_features = false
}

struct RLParameters {
  1: double gamma = 0.9,
  2: double epsilon = 0.1,
  3: double target_update_rate = 0.01,
  4: i32 reward_burnin = 1,
  5: bool maxq_learning = true,
  6: map<string, double> reward_boost,
  7: double temperature = 0.01,
  8: i32 softmax_policy = 1,
  9: bool use_seq_num_diff_as_time_diff = false,
  10: string q_network_loss = 'mse',
  11: bool set_missing_value_to_zero = false,
}

struct RainbowDQNParameters {
  1: bool double_q_learning = true,
  2: bool dueling_architecture = false,
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
  1: list<i32> layers = [-1, 512, 256, 128, 1],
  2: list<string> activations = ['relu', 'relu', 'relu', 'linear'],
}

struct FactorizationParameters {
  1: FeedForwardParameters state,
  2: FeedForwardParameters action,
}

struct TrainingParameters {
  1: i32 minibatch_size = 16384,
  2: double learning_rate = 0.01,
  3: string optimizer = 'ADAM',
  4: list<i32> layers = [-1, 512, 256, 128, 1],
  5: list<string> activations = ['relu', 'relu', 'relu', 'linear'],
  6: string lr_policy = 'fixed',
  7: double lr_decay = 0.999,
  8: double dropout_ratio = 0.0,
  9: optional string warm_start_model_path,
  10: optional CNNParameters cnn_parameters,
  11: optional FactorizationParameters factorization_parameters,
  12: double l2_decay = 0.01,
  13: bool use_noisy_linear_layers = false,
}

struct InTrainingCPEParameters {
  1: double mdp_sampled_rate,
}

struct EvolutionParameters {
  1: i32 population_size = 1000,
  2: double mutation_power = 0.1,
  3: double learning_rate = 0.01,
}

struct ActionBudget {
  1: string limited_action,
  2: double action_limit,
  3: double quantile_update_rate = 0.01,
  4: i32 quantile_update_frequency = 10,
  5: i32 window_size = 16384,
}

struct DiscreteActionModelParameters {
  1: list<string> actions,
  2: RLParameters rl,
  3: TrainingParameters training,
  4: ActionBudget action_budget,
  5: RainbowDQNParameters rainbow,
  6: optional InTrainingCPEParameters in_training_cpe,
  7: optional StateFeatureParameters state_feature_params,
  8: optional list<double> target_action_distribution,
}

struct ContinuousActionModelParameters {
  1: RLParameters rl,
  2: TrainingParameters training,
  4: RainbowDQNParameters rainbow,
  5: optional InTrainingCPEParameters in_training_cpe,
}

struct DDPGNetworkParameters {
  1: list<i32> layers = [-1, 512, 256, 128, 1],
  2: list<string> activations = ['relu', 'relu', 'relu', 'tanh'],
  3: double l2_decay = 0.01,
  4: double learning_rate = 0.001,
}

struct DDPGTrainingParameters {
  1: i32 minibatch_size = 128,
  2: double final_layer_init = 0.003,
  3: string optimizer = 'ADAM',
  4: optional string warm_start_model_path,
  5: bool use_noisy_linear_layers = false,
}

struct StateFeatureParameters {
  1: list<string> state_feature_names_override = [],
  2: list<i32> state_feature_hashes_override = [],
}

struct DDPGModelParameters {
  1: RLParameters rl,
  2: DDPGTrainingParameters shared_training,
  3: DDPGNetworkParameters actor_training,
  4: DDPGNetworkParameters critic_training,
  5: optional map<i64, list<double>> action_rescale_map = {},
  6: optional StateFeatureParameters state_feature_params,
}

struct OptimizerParameters {
  1: string optimizer = 'ADAM',
  2: double learning_rate = 0.01,
  3: double l2_decay = 0.01,
}

struct SACTrainingParameters {
  1: i32 minibatch_size = 16384,
  2: OptimizerParameters q_network_optimizer,
  3: OptimizerParameters value_network_optimizer,
  4: OptimizerParameters actor_network_optimizer,
  5: bool use_2_q_functions,
  # alpha in the paper; controlling explore & exploit
  6: double entropy_temperature = 0.1,
}

struct SACModelParameters {
  1: RLParameters rl,
  2: SACTrainingParameters training,
  3: FeedForwardParameters q_network,
  4: FeedForwardParameters value_network,
  5: FeedForwardParameters actor_network,
}

struct KNNDQNModelParameters {
  1: RLParameters rl,
  2: DDPGTrainingParameters shared_training,
  3: DDPGNetworkParameters actor_training,
  4: DDPGNetworkParameters critic_training,
  5: i64 num_actions,
  6: i32 action_dim,
  7: i64 k,
}

struct OpenAIGymParameters {
  1: i32 num_episodes = 1000,
  2: i32 max_steps = 200,
  3: i32 train_every_ts = 1,
  4: i32 train_after_ts = 1,
  5: i32 test_every_ts = 2000,
  6: i32 test_after_ts = 1,
  7: i32 num_train_batches = 1,
  8: i32 avg_over_num_episodes = 100
}
