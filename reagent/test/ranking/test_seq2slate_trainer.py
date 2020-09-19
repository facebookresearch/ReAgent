import copy
import logging
import random
import unittest

import numpy as np
import reagent.types as rlt
import torch
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union, classes
from reagent.parameters import Seq2SlateParameters
from reagent.parameters_seq2slate import IPSClamp, IPSClampMethod
from reagent.training.ranking.helper import ips_clamp
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def create_trainer(
    seq2slate_net,
    batch_size,
    learning_rate,
    device,
    seq2slate_params,
    policy_gradient_interval,
):
    use_gpu = False if device == torch.device("cpu") else True
    return Seq2SlateTrainer(
        seq2slate_net=seq2slate_net,
        minibatch_size=batch_size,
        parameters=seq2slate_params,
        policy_optimizer=Optimizer__Union(SGD=classes["SGD"](lr=learning_rate)),
        use_gpu=use_gpu,
        policy_gradient_interval=policy_gradient_interval,
        print_interval=1,
    )


def create_seq2slate_transformer(
    state_dim, candidate_num, candidate_dim, hidden_size, device
):
    return Seq2SlateTransformerNet(
        state_dim=state_dim,
        candidate_dim=candidate_dim,
        num_stacked_layers=2,
        num_heads=2,
        dim_model=hidden_size,
        dim_feedforward=hidden_size,
        max_src_seq_len=candidate_num,
        max_tgt_seq_len=candidate_num,
        encoder_only=False,
    ).to(device)


def create_on_policy_batch(
    seq2slate, batch_size, state_dim, candidate_num, candidate_dim, rank_seed, device
):
    state = torch.randn(batch_size, state_dim).to(device)
    candidates = torch.randn(batch_size, candidate_num, candidate_dim).to(device)
    reward = torch.rand(batch_size, 1).to(device)
    batch = rlt.PreprocessedRankingInput.from_input(
        state=state, candidates=candidates, device=device
    )
    # Reset seed here so that gradients can be replicated.
    torch.manual_seed(rank_seed)
    rank_output = seq2slate(
        batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=candidate_num, greedy=False
    )
    ranked_order = rank_output.ranked_tgt_out_idx - 2
    ranked_slate_prob = rank_output.ranked_per_seq_probs
    on_policy_batch = rlt.PreprocessedRankingInput.from_input(
        state=state,
        candidates=candidates,
        device=device,
        action=ranked_order,
        logged_propensities=ranked_slate_prob.detach(),
        slate_reward=reward,
    )
    return on_policy_batch


def create_off_policy_batch(
    seq2slate, batch_size, state_dim, candidate_num, candidate_dim, device
):
    state = torch.randn(batch_size, state_dim).to(device)
    candidates = torch.randn(batch_size, candidate_num, candidate_dim).to(device)
    reward = torch.rand(batch_size, 1).to(device)
    action = torch.stack(
        [torch.randperm(candidate_num).to(device) for _ in range(batch_size)]
    )
    logged_slate_prob = torch.rand(batch_size, 1).to(device) / 1e12
    off_policy_batch = rlt.PreprocessedRankingInput.from_input(
        state=state,
        candidates=candidates,
        device=device,
        action=action,
        logged_propensities=logged_slate_prob,
        slate_reward=reward,
    )
    return off_policy_batch


class TestSeq2SlateTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def assert_correct_gradient(
        self,
        net_with_gradient,
        net_after_gradient,
        policy_gradient_interval,
        learning_rate,
    ):
        for (n_c, w_c), (n, w) in zip(
            net_with_gradient.named_parameters(), net_after_gradient.named_parameters()
        ):
            assert n_c == n
            assert torch.allclose(
                w_c - policy_gradient_interval * learning_rate * w_c.grad,
                w,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_ips_clamp(self):
        importance_sampling = torch.tensor([0.5, 0.3, 3.0, 10.0, 40.0])
        assert torch.all(ips_clamp(importance_sampling, None) == importance_sampling)
        assert torch.all(
            ips_clamp(importance_sampling, IPSClamp(IPSClampMethod.AGGRESSIVE, 3.0))
            == torch.tensor([0.5, 0.3, 3.0, 0.0, 0.0])
        )
        assert torch.all(
            ips_clamp(importance_sampling, IPSClamp(IPSClampMethod.UNIVERSAL, 3.0))
            == torch.tensor([0.5, 0.3, 3.0, 3.0, 3.0])
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_trainer_on_policy_one_gradient_update_step_gpu(self):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval=1, device=torch.device("cuda")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_trainer_on_policy_multi_gradient_update_steps_gpu(self):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval=5, device=torch.device("cuda")
        )

    def test_seq2slate_trainer_on_policy_one_gradient_update_step_cpu(self):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval=1, device=torch.device("cpu")
        )

    def test_seq2slate_trainer_on_policy_multi_gradient_update_steps_cpu(self):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval=5, device=torch.device("cpu")
        )

    def _test_seq2slate_trainer_on_policy(self, policy_gradient_interval, device):
        batch_size = 32
        state_dim = 2
        candidate_num = 15
        candidate_dim = 4
        hidden_size = 16
        learning_rate = 1.0
        on_policy = True
        rank_seed = 111
        seq2slate_params = Seq2SlateParameters(on_policy=on_policy)

        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, device
        )
        seq2slate_net_copy = copy.deepcopy(seq2slate_net)
        seq2slate_net_copy_copy = copy.deepcopy(seq2slate_net)
        trainer = create_trainer(
            seq2slate_net,
            batch_size,
            learning_rate,
            device,
            seq2slate_params,
            policy_gradient_interval,
        )
        batch = create_on_policy_batch(
            seq2slate_net,
            batch_size,
            state_dim,
            candidate_num,
            candidate_dim,
            rank_seed,
            device,
        )
        for _ in range(policy_gradient_interval):
            trainer.train(rlt.PreprocessedTrainingBatch(training_input=batch))

        # manual compute gradient
        torch.manual_seed(rank_seed)
        rank_output = seq2slate_net_copy(
            batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=candidate_num, greedy=False
        )
        loss = -(
            torch.mean(torch.log(rank_output.ranked_per_seq_probs) * batch.slate_reward)
        )
        loss.backward()
        self.assert_correct_gradient(
            seq2slate_net_copy, seq2slate_net, policy_gradient_interval, learning_rate
        )

        # another way to compute gradient manually
        torch.manual_seed(rank_seed)
        ranked_per_seq_probs = seq2slate_net_copy_copy(
            batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=candidate_num, greedy=False
        ).ranked_per_seq_probs
        loss = -(
            torch.mean(
                ranked_per_seq_probs
                / ranked_per_seq_probs.detach()
                * batch.slate_reward
            )
        )
        loss.backward()
        self.assert_correct_gradient(
            seq2slate_net_copy_copy,
            seq2slate_net,
            policy_gradient_interval,
            learning_rate,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_trainer_off_policy_one_gradient_update_step_gpu(self):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval=1, device=torch.device("cuda")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_trainer_off_policy_multi_gradient_update_steps_gpu(self):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval=5, device=torch.device("cuda")
        )

    def test_seq2slate_trainer_off_policy_one_gradient_update_step_cpu(self):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval=1, device=torch.device("cpu")
        )

    def test_seq2slate_trainer_off_policy_multi_gradient_update_steps_cpu(self):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval=5, device=torch.device("cpu")
        )

    def _test_seq2slate_trainer_off_policy(self, policy_gradient_interval, device):
        batch_size = 32
        state_dim = 2
        candidate_num = 15
        candidate_dim = 4
        hidden_size = 16
        learning_rate = 1.0
        on_policy = False
        seq2slate_params = Seq2SlateParameters(on_policy=on_policy)

        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, device
        )
        seq2slate_net_copy = copy.deepcopy(seq2slate_net)
        seq2slate_net_copy_copy = copy.deepcopy(seq2slate_net)
        trainer = create_trainer(
            seq2slate_net,
            batch_size,
            learning_rate,
            device,
            seq2slate_params,
            policy_gradient_interval,
        )
        batch = create_off_policy_batch(
            seq2slate_net, batch_size, state_dim, candidate_num, candidate_dim, device
        )

        for _ in range(policy_gradient_interval):
            trainer.train(rlt.PreprocessedTrainingBatch(training_input=batch))

        # manual compute gradient
        ranked_per_seq_log_probs = seq2slate_net_copy(
            batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        ).log_probs

        loss = -(
            torch.mean(
                ranked_per_seq_log_probs
                * torch.exp(ranked_per_seq_log_probs).detach()
                / batch.tgt_out_probs
                * batch.slate_reward
            )
        )
        loss.backward()
        self.assert_correct_gradient(
            seq2slate_net_copy, seq2slate_net, policy_gradient_interval, learning_rate
        )

        # another way to compute gradient manually
        ranked_per_seq_probs = torch.exp(
            seq2slate_net_copy_copy(
                batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
            ).log_probs
        )

        loss = -(
            torch.mean(ranked_per_seq_probs / batch.tgt_out_probs * batch.slate_reward)
        )
        loss.backward()
        self.assert_correct_gradient(
            seq2slate_net_copy_copy,
            seq2slate_net,
            policy_gradient_interval,
            learning_rate,
        )

    def test_seq2slate_trainer_off_policy_with_universal_clamp(self):
        self._test_seq2slate_trainer_off_policy_with_clamp(IPSClampMethod.UNIVERSAL)

    def test_seq2slate_trainer_off_policy_with_aggressive_clamp(self):
        self._test_seq2slate_trainer_off_policy_with_clamp(IPSClampMethod.AGGRESSIVE)

    def _test_seq2slate_trainer_off_policy_with_clamp(self, clamp_method):
        batch_size = 32
        state_dim = 2
        candidate_num = 15
        candidate_dim = 4
        hidden_size = 16
        learning_rate = 1.0
        device = torch.device("cpu")
        policy_gradient_interval = 1
        seq2slate_params = Seq2SlateParameters(
            on_policy=False,
            ips_clamp=IPSClamp(clamp_method=clamp_method, clamp_max=0.3),
        )

        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, device
        )
        seq2slate_net_copy = copy.deepcopy(seq2slate_net)
        trainer = create_trainer(
            seq2slate_net,
            batch_size,
            learning_rate,
            device,
            seq2slate_params,
            policy_gradient_interval,
        )
        batch = create_off_policy_batch(
            seq2slate_net, batch_size, state_dim, candidate_num, candidate_dim, device
        )

        for _ in range(policy_gradient_interval):
            trainer.train(rlt.PreprocessedTrainingBatch(training_input=batch))

        # manual compute gradient
        ranked_per_seq_probs = torch.exp(
            seq2slate_net_copy(
                batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
            ).log_probs
        )
        logger.info(f"ips ratio={ranked_per_seq_probs / batch.tgt_out_probs}")
        loss = -(
            torch.mean(
                ips_clamp(
                    ranked_per_seq_probs / batch.tgt_out_probs,
                    seq2slate_params.ips_clamp,
                )
                * batch.slate_reward
            )
        )
        loss.backward()
        self.assert_correct_gradient(
            seq2slate_net_copy, seq2slate_net, policy_gradient_interval, learning_rate
        )
