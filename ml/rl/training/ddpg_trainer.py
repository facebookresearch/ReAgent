#!/usr/bin/env python3

from ml.rl.training.ddpg_predictor import DDPGPredictor


class DDPGTrainer(object):
    def __init__(self, parameters, env_details) -> None:
        self.actor_params = parameters.actor_training
        self.actor_minibatch_size = self.actor_params.minibatch_size
        self.minibatch_size = self.actor_minibatch_size
        self.actor_params.layers[0] = env_details.state_dim
        self.actor_params.layers[-1] = env_details.action_dim
        self.env_details = env_details

    def train(self) -> None:
        pass

    def predictor(self) -> DDPGPredictor:
        """Builds a DDPGPredictor."""
        return DDPGPredictor.export_actor(self)
