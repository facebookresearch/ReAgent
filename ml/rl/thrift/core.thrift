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
  7: double temperature = 0.5,
  8: i32 softmax_policy = 1
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

struct TrainingParameters {
  1: i32 minibatch_size = 16384,
  2: double learning_rate = 0.01,
  3: string optimizer = 'ADAM',
  4: list<i32> layers = [-1, 512, 256, 128, 1],
  5: list<string> activations = ['relu', 'relu', 'relu', 'linear'],
  6: string lr_policy = 'fixed',
  7: double gamma = 0.999,
  8: double dropout_ratio = 0.0,
  9: optional string warm_start_model_path,
  10: optional CNNParameters cnn_parameters,
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
}

struct KnnParameters {
  1: string model_type,
  2: i32 knn_frequency,
  3: i32 knn_k,
  4: bool knn_dynreindex,
  5: double knn_dynreindex_threshold,
  6: i32 knn_dynreindex_rand_other,
}

struct ContinuousActionModelParameters {
  1: RLParameters rl,
  2: TrainingParameters training,
  3: KnnParameters knn,
}

struct DDPGNetworkParameters {
  1: list<i32> layers = [-1, 512, 256, 128, 1],
  2: list<string> activations = ['relu', 'relu', 'relu', 'tanh'],
  3: double l2_decay = 0.01
  4: double learning_rate = 0.001,
}

struct DDPGTrainingParameters {
  1: i32 minibatch_size = 128,
  2: double final_layer_init = 0.003,
  3: string optimizer = 'ADAM'
}

struct DDPGModelParameters {
  1: RLParameters rl,
  2: DDPGTrainingParameters shared_training,
  3: DDPGNetworkParameters actor_training,
  4: DDPGNetworkParameters critic_training,
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
