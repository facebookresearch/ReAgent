#!/usr/bin/env python3

import torch

from ml.rl.thrift.core.ttypes import AdditionalFeatureTypes

DEFAULT_ADDITIONAL_FEATURE_TYPES = AdditionalFeatureTypes(int_features=False)


class RLTrainer:
    def __init__(self, parameters, use_gpu, additional_feature_types):

        self.minibatch = 0
        self.reward_burnin = parameters.rl.reward_burnin
        self._additional_feature_types = additional_feature_types
        self.rl_temperature = parameters.rl.temperature
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.use_seq_num_diff_as_time_diff = parameters.rl.use_seq_num_diff_as_time_diff

        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
            self.dtype = torch.cuda.FloatTensor
            self.dtypelong = torch.cuda.LongTensor
        else:
            self.use_gpu = False
            self.dtype = torch.FloatTensor
            self.dtypelong = torch.LongTensor

    def _set_optimizer(self, optimizer_name):
        if optimizer_name == "ADAM":
            self.optimizer_func = torch.optim.Adam
        elif optimizer_name == "SGD":
            self.optimizer_func = torch.optim.SGD
        else:
            raise NotImplementedError(
                "{} optimizer not implemented".format(optimizer_name)
            )

    def _soft_update(self, network, target_network, tau) -> None:
        """ Target network update logic as defined in DDPG paper
        updated_params = tau * network_params + (1 - tau) * target_network_params
        :param network network with parameters to include in soft update
        :param target_network target network with params to soft update
        :param tau hyperparameter to control target tracking speed
        """
        for t_param, param in zip(target_network.parameters(), network.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)

    def train(self, training_samples, evaluator=None, episode_values=None) -> None:
        raise NotImplementedError()
