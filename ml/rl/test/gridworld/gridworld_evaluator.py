#!/usr/bin/env python3


class GridworldEvaluator(object):
    def __init__(
        self,
        env,
        assume_optimal_policy: bool,
        use_int_features: bool = False,
    ) -> None:
        self._test_states, self._test_actions, _, _, _, _, _, _ = env.generate_samples(
            100000, 1.0
        )
        self._test_values = env.true_values_for_sample(
            self._test_states, self._test_actions, assume_optimal_policy
        )
        self._env = env
        self.use_int_features = use_int_features

    def _split_int_and_float_features(self, features):
        float_features, int_features = [], []
        for example in features:
            float_dict, int_dict = {}, {}
            for k, v in example.items():
                if isinstance(v, int):
                    int_dict[k] = v
                else:
                    float_dict[k] = v
            float_features.append(float_dict)
            int_features.append(int_dict)
        return float_features, int_features

    def evaluate(self, predictor):
        # Test feeding float features & int features
        if self.use_int_features:
            float_features, int_features = self._split_int_and_float_features(
                self._test_states
            )
            # Since all gridworld features are float types, swap these so
            # all inputs are now int_features for testing purpose
            float_features, int_features = int_features, float_features
            prediction = predictor.predict(float_features, int_features)
        # Test only feeding float features
        else:
            prediction = predictor.predict(self._test_states)

        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction[x][self._test_actions[x]]
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))


class GridworldContinuousEvaluator(GridworldEvaluator):
    def evaluate(self, predictor):
        # Test feeding float features & int features
        if self.use_int_features:
            float_features, int_features = self._split_int_and_float_features(
                self._test_states
            )
            # Since all gridworld features are float types, swap these so
            # all inputs are now int_features for testing purpose
            float_features, int_features = int_features, float_features
            prediction = predictor.predict(
                float_state_features=float_features,
                int_state_features=int_features,
                actions=self._test_actions,
            )
        # Test only feeding float features
        else:
            prediction = predictor.predict(
                float_state_features=self._test_states,
                int_state_features=None,
                actions=self._test_actions,
            )
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction[x]['Q']
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))


class GridworldDDPGEvaluator(GridworldEvaluator):
    def evaluate_actor(self, actor):
        actor_prediction = actor.actor_prediction(self._test_states)
        print(
            'Actor predictions executed successfully. Sample: {}'.
            format(actor_prediction)
        )

    def evaluate_critic(self, critic):
        critic_prediction = critic.critic_prediction(
            float_state_features=self._test_states,
            int_state_features=None,
            actions=self._test_actions,
        )
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = critic_prediction[x]
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))
