#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import numpy as np
import torch
from ml.rl.models.mdn_rnn import MDNRNNMemoryPool, gmm_loss
from ml.rl.models.world_model import MemoryNetwork
from ml.rl.test.world_model.simulated_world_model import SimulatedWorldModel
from ml.rl.thrift.core.ttypes import MDNRNNParameters
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


logger = logging.getLogger(__name__)


class TestMDNRNN(unittest.TestCase):
    def test_gmm_loss(self):
        # seq_len x batch_size x gaussian_size x feature_size
        # 1 x 1 x 2 x 2
        mus = torch.Tensor([[[[0.0, 0.0], [6.0, 6.0]]]])
        sigmas = torch.Tensor([[[[2.0, 2.0], [2.0, 2.0]]]])
        # seq_len x batch_size x gaussian_size
        pi = torch.Tensor([[[0.5, 0.5]]])
        logpi = torch.log(pi)

        # seq_len x batch_size x feature_size
        batch = torch.Tensor([[[3.0, 3.0]]])
        gl = gmm_loss(batch, mus, sigmas, logpi)

        # first component, first dimension
        n11 = Normal(mus[0, 0, 0, 0], sigmas[0, 0, 0, 0])
        # first component, second dimension
        n12 = Normal(mus[0, 0, 0, 1], sigmas[0, 0, 0, 1])
        p1 = (
            pi[0, 0, 0]
            * torch.exp(n11.log_prob(batch[0, 0, 0]))
            * torch.exp(n12.log_prob(batch[0, 0, 1]))
        )
        # second component, first dimension
        n21 = Normal(mus[0, 0, 1, 0], sigmas[0, 0, 1, 0])
        # second component, second dimension
        n22 = Normal(mus[0, 0, 1, 1], sigmas[0, 0, 1, 1])
        p2 = (
            pi[0, 0, 1]
            * torch.exp(n21.log_prob(batch[0, 0, 0]))
            * torch.exp(n22.log_prob(batch[0, 0, 1]))
        )

        logger.info(
            "gmm loss={}, p1={}, p2={}, p1+p2={}, -log(p1+p2)={}".format(
                gl, p1, p2, p1 + p2, -torch.log(p1 + p2)
            )
        )
        assert -torch.log(p1 + p2) == gl

    def test_mdnrnn_simulate_world_cpu(self):
        self._test_mdnrnn_simulate_world()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mdnrnn_simulate_world_gpu(self):
        self._test_mdnrnn_simulate_world(use_gpu=True)

    def _test_mdnrnn_simulate_world(self, use_gpu=False):
        num_epochs = 300
        num_episodes = 400
        batch_size = 200
        action_dim = 2
        seq_len = 5
        state_dim = 2
        simulated_num_gaussians = 2
        mdrnn_num_gaussians = 2
        simulated_num_hidden_layers = 1
        simulated_num_hiddens = 3
        mdnrnn_num_hidden_layers = 1
        mdnrnn_num_hiddens = 10
        adam_lr = 0.01

        replay_buffer = MDNRNNMemoryPool(max_replay_memory_size=num_episodes)
        swm = SimulatedWorldModel(
            action_dim=action_dim,
            state_dim=state_dim,
            num_gaussians=simulated_num_gaussians,
            lstm_num_hidden_layers=simulated_num_hidden_layers,
            lstm_num_hiddens=simulated_num_hiddens,
        )

        possible_actions = torch.eye(action_dim)
        for _ in range(num_episodes):
            cur_state_mem = np.zeros((seq_len, state_dim))
            next_state_mem = np.zeros((seq_len, state_dim))
            action_mem = np.zeros((seq_len, action_dim))
            reward_mem = np.zeros(seq_len)
            not_terminal_mem = np.zeros(seq_len)
            next_mus_mem = np.zeros((seq_len, simulated_num_gaussians, state_dim))

            swm.init_hidden(batch_size=1)
            next_state = torch.randn((1, 1, state_dim))
            for s in range(seq_len):
                cur_state = next_state
                action = possible_actions[np.random.randint(action_dim)].view(
                    1, 1, action_dim
                )
                next_mus, reward = swm(action, cur_state)

                not_terminal = 1
                if s == seq_len - 1:
                    not_terminal = 0

                # randomly draw for next state
                next_pi = torch.ones(simulated_num_gaussians) / simulated_num_gaussians
                index = Categorical(next_pi).sample((1,)).long().item()
                next_state = next_mus[0, 0, index].view(1, 1, state_dim)

                cur_state_mem[s] = cur_state.detach().numpy()
                action_mem[s] = action.numpy()
                reward_mem[s] = reward.detach().numpy()
                not_terminal_mem[s] = not_terminal
                next_state_mem[s] = next_state.detach().numpy()
                next_mus_mem[s] = next_mus.detach().numpy()

            replay_buffer.insert_into_memory(
                cur_state_mem, action_mem, next_state_mem, reward_mem, not_terminal_mem
            )

        num_batch = num_episodes // batch_size
        mdnrnn_params = MDNRNNParameters(
            hidden_size=mdnrnn_num_hiddens,
            num_hidden_layers=mdnrnn_num_hidden_layers,
            minibatch_size=batch_size,
            learning_rate=adam_lr,
            num_gaussians=mdrnn_num_gaussians,
        )
        mdnrnn_net = MemoryNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_hiddens=mdnrnn_params.hidden_size,
            num_hidden_layers=mdnrnn_params.num_hidden_layers,
            num_gaussians=mdnrnn_params.num_gaussians,
        )
        if use_gpu and torch.cuda.is_available():
            mdnrnn_net = mdnrnn_net.cuda()
        trainer = MDNRNNTrainer(
            mdnrnn_network=mdnrnn_net, params=mdnrnn_params, cum_loss_hist=num_batch
        )

        for e in range(num_epochs):
            for i in range(num_batch):
                training_batch = replay_buffer.sample_memories(
                    batch_size, use_gpu=use_gpu, batch_first=use_gpu
                )
                losses = trainer.train(training_batch, batch_first=use_gpu)
                logger.info(
                    "{}-th epoch, {}-th minibatch: \n"
                    "loss={}, bce={}, gmm={}, mse={} \n"
                    "cum loss={}, cum bce={}, cum gmm={}, cum mse={}\n".format(
                        e,
                        i,
                        losses["loss"],
                        losses["bce"],
                        losses["gmm"],
                        losses["mse"],
                        np.mean(trainer.cum_loss),
                        np.mean(trainer.cum_bce),
                        np.mean(trainer.cum_gmm),
                        np.mean(trainer.cum_mse),
                    )
                )

                if (
                    np.mean(trainer.cum_loss) < 0
                    and np.mean(trainer.cum_gmm) < -3.0
                    and np.mean(trainer.cum_bce) < 0.6
                    and np.mean(trainer.cum_mse) < 0.2
                ):
                    return

        assert False, "losses not reduced significantly during training"
