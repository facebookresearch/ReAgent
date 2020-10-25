import copy
import itertools
import logging
import random
import unittest
from itertools import permutations

import numpy as np
import numpy.testing as npt
import reagent.types as rlt
import torch
from parameterized import parameterized
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union, classes
from reagent.parameters import Seq2SlateParameters
from reagent.parameters_seq2slate import IPSClamp, IPSClampMethod
from reagent.samplers.frechet import FrechetSort
from reagent.training.ranking.helper import ips_clamp
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


output_arch_list = [
    Seq2SlateOutputArch.FRECHET_SORT,
    Seq2SlateOutputArch.AUTOREGRESSIVE,
]
policy_gradient_interval_list = [1, 5]
clamp_method_list = [IPSClampMethod.UNIVERSAL, IPSClampMethod.UNIVERSAL]
clamp_max_list = [1.0, 10.0]
frechet_sort_shape_list = [0.1, 0.5, 1.0]


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
    state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
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
        output_arch=output_arch,
        temperature=0.5,
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
                atol=2e-6,
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
    @parameterized.expand(
        itertools.product(policy_gradient_interval_list, output_arch_list)
    )
    def test_seq2slate_trainer_on_policy_gpu(
        self, policy_gradient_interval, output_arch
    ):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval, output_arch, device=torch.device("cuda")
        )

    @parameterized.expand(
        itertools.product(policy_gradient_interval_list, output_arch_list)
    )
    def test_seq2slate_trainer_on_policy_cpu(
        self, policy_gradient_interval, output_arch
    ):
        self._test_seq2slate_trainer_on_policy(
            policy_gradient_interval, output_arch, device=torch.device("cpu")
        )

    def _test_seq2slate_trainer_on_policy(
        self, policy_gradient_interval, output_arch, device
    ):
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
            state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
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
    @parameterized.expand(
        itertools.product(policy_gradient_interval_list, output_arch_list)
    )
    def test_seq2slate_trainer_off_policy_gpu(
        self, policy_gradient_interval, output_arch
    ):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval, output_arch, device=torch.device("cuda")
        )

    @parameterized.expand(
        itertools.product(policy_gradient_interval_list, output_arch_list)
    )
    def test_seq2slate_trainer_off_policy_cpu(
        self, policy_gradient_interval, output_arch
    ):
        self._test_seq2slate_trainer_off_policy(
            policy_gradient_interval, output_arch, device=torch.device("cpu")
        )

    def _test_seq2slate_trainer_off_policy(
        self, policy_gradient_interval, output_arch, device
    ):
        batch_size = 32
        state_dim = 2
        candidate_num = 15
        candidate_dim = 4
        hidden_size = 16
        learning_rate = 1.0
        on_policy = False
        seq2slate_params = Seq2SlateParameters(on_policy=on_policy)

        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
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

    @parameterized.expand(itertools.product(clamp_method_list, output_arch_list))
    def test_seq2slate_trainer_off_policy_with_clamp(self, clamp_method, output_arch):
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
            state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
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

    @parameterized.expand(
        itertools.product(
            output_arch_list, clamp_method_list, clamp_max_list, frechet_sort_shape_list
        )
    )
    def test_compute_impt_smpl(self, output_arch, clamp_method, clamp_max, shape):
        logger.info(f"output arch: {output_arch}")
        logger.info(f"clamp method: {clamp_method}")
        logger.info(f"clamp max: {clamp_max}")
        logger.info(f"frechet shape: {shape}")

        candidate_num = 5
        candidate_dim = 2
        state_dim = 1
        hidden_size = 32
        device = torch.device("cpu")
        batch_size = 32
        learning_rate = 0.001
        policy_gradient_interval = 1

        candidates = torch.randint(5, (candidate_num, candidate_dim)).float()
        candidate_scores = torch.sum(candidates, dim=1)

        seq2slate_params = Seq2SlateParameters(
            on_policy=False,
            ips_clamp=IPSClamp(clamp_method=clamp_method, clamp_max=clamp_max),
        )
        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
        )
        trainer = create_trainer(
            seq2slate_net,
            batch_size,
            learning_rate,
            device,
            seq2slate_params,
            policy_gradient_interval,
        )

        all_permt = torch.tensor(
            list(permutations(range(candidate_num), candidate_num))
        )
        sampler = FrechetSort(shape=shape, topk=candidate_num)
        sum_of_logged_propensity = 0
        sum_of_model_propensity = 0
        sum_of_ips_ratio = 0

        for i in range(len(all_permt)):
            sample_action = all_permt[i]
            logged_propensity = torch.exp(
                sampler.log_prob(candidate_scores, sample_action)
            )
            batch = rlt.PreprocessedRankingInput.from_input(
                state=torch.zeros(1, state_dim),
                candidates=candidates.unsqueeze(0),
                device=device,
                action=sample_action.unsqueeze(0),
                logged_propensities=logged_propensity.reshape(1, 1),
            )
            model_propensities = torch.exp(
                seq2slate_net(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probs
            )
            impt_smpl, clamped_impt_smpl = trainer._compute_impt_smpl(
                model_propensities, logged_propensity
            )
            if impt_smpl > clamp_max:
                if clamp_method == IPSClampMethod.AGGRESSIVE:
                    npt.asset_allclose(clamped_impt_smpl.detach().numpy(), 0, rtol=1e-5)
                else:
                    npt.assert_allclose(
                        clamped_impt_smpl.detach().numpy(), clamp_max, rtol=1e-5
                    )

            sum_of_model_propensity += model_propensities
            sum_of_logged_propensity += logged_propensity
            sum_of_ips_ratio += model_propensities / logged_propensity
            logger.info(
                f"shape={shape}, sample_action={sample_action}, logged_propensity={logged_propensity},"
                f" model_propensity={model_propensities}"
            )

        logger.info(
            f"shape {shape}, sum_of_logged_propensity={sum_of_logged_propensity}, "
            f"sum_of_model_propensity={sum_of_model_propensity}, "
            f"mean sum_of_ips_ratio={sum_of_ips_ratio / len(all_permt)}"
        )
        npt.assert_allclose(sum_of_logged_propensity.detach().numpy(), 1, rtol=1e-5)
        npt.assert_allclose(sum_of_model_propensity.detach().numpy(), 1, rtol=1e-5)

    @parameterized.expand(itertools.product(output_arch_list, frechet_sort_shape_list))
    def test_ips_ratio_mean(self, output_arch, shape):
        output_arch = Seq2SlateOutputArch.FRECHET_SORT
        shape = 0.1
        logger.info(f"output arch: {output_arch}")
        logger.info(f"frechet shape: {shape}")

        candidate_num = 5
        candidate_dim = 2
        state_dim = 1
        hidden_size = 8
        device = torch.device("cpu")
        batch_size = 1024
        num_batches = 400
        learning_rate = 0.001
        policy_gradient_interval = 1

        state = torch.zeros(batch_size, state_dim)
        # all data have same candidates
        candidates = torch.randint(
            5, (batch_size, candidate_num, candidate_dim)
        ).float()
        candidates[1:] = candidates[0]
        candidate_scores = torch.sum(candidates, dim=-1)

        seq2slate_params = Seq2SlateParameters(
            on_policy=False,
        )
        seq2slate_net = create_seq2slate_transformer(
            state_dim, candidate_num, candidate_dim, hidden_size, output_arch, device
        )
        trainer = create_trainer(
            seq2slate_net,
            batch_size,
            learning_rate,
            device,
            seq2slate_params,
            policy_gradient_interval,
        )

        sampler = FrechetSort(shape=shape, topk=candidate_num)
        sum_of_ips_ratio = 0

        for i in range(num_batches):
            sample_outputs = [
                sampler.sample_action(candidate_scores[j : j + 1])
                for j in range(batch_size)
            ]
            action = torch.stack(
                list(map(lambda x: x.action.squeeze(0), sample_outputs))
            )
            logged_propensity = torch.stack(
                list(map(lambda x: torch.exp(x.log_prob), sample_outputs))
            )
            batch = rlt.PreprocessedRankingInput.from_input(
                state=state,
                candidates=candidates,
                device=device,
                action=action,
                logged_propensities=logged_propensity,
            )
            model_propensities = torch.exp(
                seq2slate_net(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probs
            )
            impt_smpl, _ = trainer._compute_impt_smpl(
                model_propensities, logged_propensity
            )
            sum_of_ips_ratio += torch.mean(impt_smpl).detach().numpy()
            mean_of_ips_ratio = sum_of_ips_ratio / (i + 1)
            logger.info(f"{i}-th batch, mean ips ratio={mean_of_ips_ratio}")

            if i > 100 and np.allclose(mean_of_ips_ratio, 1, atol=0.03):
                return

        raise Exception(f"Mean ips ratio {mean_of_ips_ratio} is not close to 1")
