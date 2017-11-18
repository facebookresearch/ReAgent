namespace py ml.rl.thrift.core

struct RLParameters {
  1: double gamma = 0.9,
  2: double epsilon = 0.1,
  3: double target_update_rate = 0.01,
  4: i32 reward_burnin = 1,
  5: bool maxq_learning = true,
}

struct TrainingParameters {
  1: i32 minibatch_size = 16384,
  2: double learning_rate = 0.01,
  3: string optimizer = 'ADAM',
  4: list<i32> layers = [-1, 512, 256, 128, 1],
  5: list<string> activations = ['relu', 'relu', 'relu', 'linear']
  6: string lr_policy = 'fixed',
  7: double gamma = 0.999,
  8: double dropout_ratio = 0.0,
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
  2: RLParameters rl = {},
  3: TrainingParameters training = {},
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
  1: RLParameters rl = {},
  2: TrainingParameters training = {},
  3: KnnParameters knn,
}
