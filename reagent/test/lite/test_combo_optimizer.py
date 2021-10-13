#!/usr/bin/env python3

import random
import unittest
from collections import defaultdict
from typing import Dict
from unittest.mock import Mock

import nevergrad as ng
import numpy as np
import torch
import torch.nn as nn
from reagent.lite.optimizer import (
    PolicyGradientOptimizer,
    GumbelSoftmaxOptimizer,
    QLearningOptimizer,
    NeverGradOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GREEDY_TEMP,
    sol_to_tensors,
)

# nevergrad performs a little worse in the test environment
NEVERGRAD_TEST_THRES = 6.0
POLICY_GRADIENT_TEST_THRES = 3.0
GUMBEL_SOFTMAX_TEST_THRES = 3.0
Q_LEARNING_TEST_THRES = 3.0


class GroundTruthNet(nn.Module):
    def __init__(self, dim_input, dim_model):
        super().__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(dim_input, dim_model),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_model, 1),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, -3, 3)

    def forward(self, x):
        return self.net(x)


def random_sample(input_param, obj_func, n_generations=100):
    """Return the best result from random sampling"""
    rs_optimizer = RandomSearchOptimizer(
        input_param,
        obj_func,
        batch_size=512,
    )
    min_reward_rs_optimizer = torch.tensor(9999.0)
    print("Random Sampling")
    for i in range(n_generations):
        (
            sampled_solutions,
            reward,
        ) = rs_optimizer.optimize_step()
        min_reward_rs_optimizer = torch.min(
            min_reward_rs_optimizer, torch.min(reward.data)
        )
        print(f"Generation={i}, min_reward={min_reward_rs_optimizer}")
    print()

    return min_reward_rs_optimizer


def discrete_input_param():
    # Some random discrete choice space
    ng_param = ng.p.Dict(
        choice1=ng.p.Choice(["128", "256", "512", "768"]),
        choice2=ng.p.Choice(["128", "256", "512", "768"]),
        choice3=ng.p.Choice(["True", "False"]),
        choice4=ng.p.Choice(["Red", "Blue", "Green", "Yellow", "Purple"]),
        choice5=ng.p.Choice(["Red", "Blue", "Green", "Yellow", "Purple"]),
    )
    return ng_param


def create_ground_truth_net(ng_param):
    dim_input = sum([len(ng_param[k].choices) for k in ng_param])
    dim_model = 256
    gt_net = GroundTruthNet(dim_input, dim_model)
    print(f"Ground-Truth Net DIM_INPUT={dim_input}, DIM_MODEL={dim_model}")
    return gt_net


def create_discrete_choice_obj_func(ng_param, gt_net):
    def obj_func(sampled_sol: Dict[str, torch.Tensor]) -> torch.Tensor:
        # sampled_sol format:
        #    key = choice_name
        #    val = choice_idx (a tensor of length `batch_size`)
        assert list(sampled_sol.values())[0].dim() == 1
        batch_size = list(sampled_sol.values())[0].shape[0]
        batch_tensors = []
        for i in range(batch_size):
            tensors = []
            for k in sorted(sampled_sol.keys()):
                num_choices = len(ng_param[k].choices)
                one_hot = torch.zeros(num_choices)
                one_hot[sampled_sol[k][i]] = 1
                tensors.append(one_hot)
            batch_tensors.append(torch.cat(tensors, dim=-1))
        batch_tensors = torch.stack(batch_tensors)
        return gt_net(batch_tensors)

    return obj_func


def create_discrete_choice_gumbel_softmax_obj_func(ng_param, gt_net):
    def obj_func(sampled_sol: Dict[str, torch.Tensor]) -> torch.Tensor:
        # sampled_sol format:
        #    key = choice_name
        #    val = sampled softmax distribution, a tensor of shape (batch_size, num_choices)
        assert list(sampled_sol.values())[0].dim() == 2
        batch_size = list(sampled_sol.values())[0].shape[0]
        batch_tensors = []
        for i in range(batch_size):
            tensors = []
            for k in sorted(sampled_sol.keys()):
                tensors.append(sampled_sol[k][i])
            batch_tensors.append(torch.cat(tensors, dim=-1))
        batch_tensors = torch.stack(batch_tensors)
        return gt_net(batch_tensors)

    return obj_func


class TestComboOptimizer(unittest.TestCase):
    def setUp(self):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_random_sample_with_raw_choices_using_uncommon_key(self):
        batch_size = 200
        input_param = ng.p.Dict(
            **{
                "#1": ng.p.Choice([32, 64, 128]),
                "choice2[3]": ng.p.Choice([True, False]),
                "choice3.attr": ng.p.Choice(
                    ["Red", "Blue", "Green", "Yellow", "Purple"]
                ),
            }
        )
        obj_func = None

        sampling_weights = {
            "#1": [0.5, 0.5, 0.0],
            "choice2[3]": [0.25, 0.75],
            "choice3.attr": [0.1, 0.9, 0.0, 0.0, 0.0],
        }

        optimizer = RandomSearchOptimizer(
            input_param,
            obj_func,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
        )
        sampled_sol = optimizer.sample(batch_size)
        sampled_sol = optimizer.indices_to_raw_choices(sampled_sol)
        self.assertEqual(len(sampled_sol), batch_size)
        self.assertIsInstance(sampled_sol, list)

        counts = {key: defaultdict(int) for key in sampling_weights}
        for sample in sampled_sol:
            self.assertSetEqual(set(sample.keys()), set(input_param.keys()))
            self.assertIn(sample["#1"], [32, 64])
            self.assertIn(sample["choice2[3]"], [True, False])
            self.assertIn(sample["choice3.attr"], ["Red", "Blue"])
            for key in sample:
                counts[key][sample[key]] += 1

        self.assertAlmostEqual(counts["#1"][32] / float(batch_size), 0.5, places=1)
        self.assertAlmostEqual(counts["#1"][64] / float(batch_size), 0.5, places=1)
        self.assertEqual(counts["#1"][128], 0)

        self.assertAlmostEqual(
            counts["choice2[3]"][True] / float(batch_size), 0.25, places=1
        )
        self.assertAlmostEqual(
            counts["choice2[3]"][False] / float(batch_size), 0.75, places=1
        )

        self.assertAlmostEqual(
            counts["choice3.attr"]["Red"] / float(batch_size), 0.1, places=1
        )
        self.assertAlmostEqual(
            counts["choice3.attr"]["Blue"] / float(batch_size), 0.9, places=1
        )
        self.assertEqual(counts["choice3.attr"]["Green"], 0)
        self.assertEqual(counts["choice3.attr"]["Yellow"], 0)
        self.assertEqual(counts["choice3.attr"]["Purple"], 0)

    def test_random_sample_with_raw_choices_1(self):
        batch_size = 1
        input_param = ng.p.Dict(
            choice1=ng.p.Choice([32, 64, 128]),
            choice2=ng.p.Choice([True, False]),
            choice3=ng.p.Choice(["Red", "Blue", "Green", "Yellow", "Purple"]),
        )
        obj_func = None
        optimizer = RandomSearchOptimizer(
            input_param, obj_func, batch_size=batch_size, sampling_weights=None
        )
        sampled_sol = optimizer.sample(batch_size)
        sampled_sol = optimizer.indices_to_raw_choices(sampled_sol)
        self.assertEqual(len(sampled_sol), batch_size)
        self.assertIsInstance(sampled_sol, list)
        for sample in sampled_sol:
            self.assertSetEqual(set(sample.keys()), set(input_param.keys()))
            for key in sample:
                self.assertIn(sample[key], input_param[key].choices.value)

    def test_random_sample_with_raw_choices_2(self):
        batch_size = 200
        input_param = ng.p.Dict(
            choice1=ng.p.Choice([32, 64, 128]),
            choice2=ng.p.Choice([True, False]),
            choice3=ng.p.Choice(["Red", "Blue", "Green", "Yellow", "Purple"]),
        )
        obj_func = None

        sampling_weights = {
            "choice1": [0.5, 0.5, 0.0],
            "choice2": [0.25, 0.75],
            "choice3": [0.1, 0.9, 0.0, 0.0, 0.0],
        }

        optimizer = RandomSearchOptimizer(
            input_param,
            obj_func,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
        )
        sampled_sol = optimizer.sample(batch_size)
        sampled_sol = optimizer.indices_to_raw_choices(sampled_sol)
        self.assertEqual(len(sampled_sol), batch_size)
        self.assertIsInstance(sampled_sol, list)

        counts = {key: defaultdict(int) for key in sampling_weights}
        for sample in sampled_sol:
            self.assertSetEqual(set(sample.keys()), set(input_param.keys()))
            self.assertIn(sample["choice1"], [32, 64])
            self.assertIn(sample["choice2"], [True, False])
            self.assertIn(sample["choice3"], ["Red", "Blue"])
            for key in sample:
                counts[key][sample[key]] += 1

        self.assertAlmostEqual(counts["choice1"][32] / float(batch_size), 0.5, places=1)
        self.assertAlmostEqual(counts["choice1"][64] / float(batch_size), 0.5, places=1)
        self.assertEqual(counts["choice1"][128], 0)

        self.assertAlmostEqual(
            counts["choice2"][True] / float(batch_size), 0.25, places=1
        )
        self.assertAlmostEqual(
            counts["choice2"][False] / float(batch_size), 0.75, places=1
        )

        self.assertAlmostEqual(
            counts["choice3"]["Red"] / float(batch_size), 0.1, places=1
        )
        self.assertAlmostEqual(
            counts["choice3"]["Blue"] / float(batch_size), 0.9, places=1
        )
        self.assertEqual(counts["choice3"]["Green"], 0)
        self.assertEqual(counts["choice3"]["Yellow"], 0)
        self.assertEqual(counts["choice3"]["Purple"], 0)

    def test_nevergrad_optimizer_discrete(self):
        batch_size = 32
        n_generations = 40
        input_param = discrete_input_param()
        gt_net = create_ground_truth_net(input_param)
        obj_func = create_discrete_choice_obj_func(input_param, gt_net)
        optimizer = NeverGradOptimizer(
            input_param,
            batch_size * n_generations,  # estimated_budgets
            obj_func=obj_func,
            batch_size=batch_size,
            optimizer_name="DoubleFastGADiscreteOnePlusOne",
        )
        best_rs_result = random_sample(input_param, obj_func, n_generations=20)
        history_min_reward = torch.tensor(9999.0)
        for i in range(n_generations):
            (
                sampled_solutions,
                reward,
            ) = optimizer.optimize_step()
            history_min_reward = torch.min(history_min_reward, torch.min(reward.data))
            print(
                f"Generation={i}, min_reward={torch.min(reward.data)}, "
                f"history_min_reward={history_min_reward}"
            )
        assert (
            abs(best_rs_result - history_min_reward) < NEVERGRAD_TEST_THRES
        ), f"Learning not converged. best random search={best_rs_result}, optimizer best result={history_min_reward}"
        assert (
            optimizer.best_solutions(1)[0][0] == history_min_reward
        ), "Best solutions (n=1) inconsistent with the best reward"
        # just test sampling() can run
        optimizer.sample(10)

    def test_policy_gradient_optimizer_discrete(self):
        batch_size = 32
        learning_rate = 0.1
        input_param = discrete_input_param()
        gt_net = create_ground_truth_net(input_param)
        obj_func = create_discrete_choice_obj_func(input_param, gt_net)
        optimizer = PolicyGradientOptimizer(
            input_param, obj_func, batch_size=batch_size, learning_rate=learning_rate
        )
        best_rs_result = random_sample(input_param, obj_func, n_generations=20)
        n_generations = 100
        for i in range(n_generations):
            (
                sampled_solutions,
                reward,
                sampled_log_probs,
            ) = optimizer.optimize_step()
            mean_reward = torch.mean(reward.data)
            print(
                f"Generation={i}, mean_reward={mean_reward}, "
                f"min_reward={torch.min(reward.data)}, "
                f"mean_sample_prob={torch.mean(torch.exp(sampled_log_probs))}, "
                f"temperature={optimizer.temp}"
            )
        assert (
            abs(best_rs_result - mean_reward) < POLICY_GRADIENT_TEST_THRES
        ), f"Learning not converged. best random search={best_rs_result}, optimizer mean result={mean_reward}"
        # just test sampling() can run
        optimizer.sample(10)

    def test_q_learning_optimizer_discrete(self):
        batch_size = 256
        input_param = discrete_input_param()
        gt_net = create_ground_truth_net(input_param)
        obj_func = create_discrete_choice_obj_func(input_param, gt_net)
        optimizer = QLearningOptimizer(input_param, obj_func, batch_size=batch_size)
        best_rs_result = random_sample(input_param, obj_func, n_generations=20)
        n_generations = 100
        for i in range(n_generations):
            (
                sampled_solutions,
                reward,
            ) = optimizer.optimize_step()
            mean_reward = torch.mean(reward.data)
            print(
                f"Generation={i}, mean_reward={mean_reward}, "
                f"min_reward={torch.min(reward.data)}, "
                f"temperature={optimizer.temp}"
            )

        eval_result = obj_func(optimizer.sample(1))
        assert (
            abs(best_rs_result - eval_result) < Q_LEARNING_TEST_THRES
        ), f"Learning not converged. best random search={best_rs_result}, eval result={eval_result}"

    def test_gumbel_softmax_optimizer_discrete(self):
        batch_size = 32
        anneal_rate = 0.97
        learning_rate = 0.1
        input_param = discrete_input_param()
        gt_net = create_ground_truth_net(input_param)
        obj_func = create_discrete_choice_gumbel_softmax_obj_func(input_param, gt_net)
        optimizer = GumbelSoftmaxOptimizer(
            input_param,
            obj_func,
            anneal_rate=anneal_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        obj_func_rs = create_discrete_choice_obj_func(input_param, gt_net)
        best_rs_result = random_sample(input_param, obj_func_rs, n_generations=20)

        n_generations = 100
        for i in range(n_generations):
            (sampled_softmax_vals, reward, logits) = optimizer.optimize_step()
            mean_reward = torch.mean(reward.data)
            print(
                f"Generation={i}, mean_reward={mean_reward}, "
                f"min_reward={torch.min(reward.data)}, "
                f"temperature={optimizer.temp}"
            )
        assert (
            optimizer.temp == optimizer.min_temp
        ), "Towards the end of learning, GumbelSoftmax Optimizer should have a low temperature"
        assert (
            abs(best_rs_result - mean_reward) < GUMBEL_SOFTMAX_TEST_THRES
        ), f"Learning not converged. best random search={best_rs_result}, optimizer mean result={mean_reward}"
        eval_obj_func = create_discrete_choice_obj_func(input_param, gt_net)
        eval_result = eval_obj_func(optimizer.sample(1))
        assert (
            abs(best_rs_result - eval_result) < GUMBEL_SOFTMAX_TEST_THRES
        ), f"Learning not converged. best random search={best_rs_result}, eval result={eval_result}"

    def run_policy_gradient_optimizer(
        self,
        input_param,
        obj_func,
        batch_size,
        n_generations,
        repeats,
    ):
        results = []
        for r in range(repeats):
            print(f"\n\n**** Policy Gradient Optimizer, Repeat={r} ****")
            pg_optimizer = PolicyGradientOptimizer(
                input_param,
                obj_func,
                batch_size=batch_size,
            )
            for i in range(n_generations):
                # non-exploration at the last generation
                if i == n_generations - 1:
                    pg_optimizer.temp = GREEDY_TEMP
                temp = pg_optimizer.temp
                (
                    sampled_solutions,
                    reward,
                    sampled_log_probs,
                ) = pg_optimizer.optimize_step()
                mean_reward_pg_optimizer = torch.mean(reward.data)
                min_reward_pg_optimizer = torch.min(reward.data)
                print(
                    f"Generation={i}, mean_reward={mean_reward_pg_optimizer}, "
                    f"min_reward={min_reward_pg_optimizer}, "
                    f"mean_sample_prob={torch.mean(torch.exp(sampled_log_probs))}, "
                    f"temperature={temp}"
                )
            results.append(mean_reward_pg_optimizer)

        return results

    def run_q_learning_optimizer(
        self,
        input_param,
        obj_func,
        batch_size,
        n_generations,
        repeats,
    ):
        results = []
        for r in range(repeats):
            print(f"\n\n**** QLearning Optimizer, Repeat={r} ****")
            ql_optimizer = QLearningOptimizer(
                input_param,
                obj_func,
                batch_size=batch_size,
                anneal_rate=0.997,
            )
            for i in range(n_generations):
                # non-exploration at the last generation
                if i == n_generations - 1:
                    ql_optimizer.temp = GREEDY_TEMP

                temp = ql_optimizer.temp
                (
                    sampled_solutions,
                    reward,
                ) = ql_optimizer.optimize_step()
                mean_reward_ql_optimizer = torch.mean(reward.data)
                min_reward_ql_optimizer = torch.min(reward.data)
                print(
                    f"Generation={i}, mean_reward={mean_reward_ql_optimizer}, "
                    f"min_reward={min_reward_ql_optimizer}, "
                    f"temp={temp}"
                )
            results.append(mean_reward_ql_optimizer)

        return results

    def test_policy_gradient_vs_q_learning_discrete(self):
        """
        Comparison between policy gradient and Q-learning-based optimizer
        The input param has two axes, choice1 and choice2.

        The value achieved by different combinations of the two choices:
              a     b     c
        1   0.43   0.9   0.45

        2   0.9    0.4    0.9

        3   0.45   0.9   0.45

        In summary, the global minimum is at (choice1=2, choice2=b), but there are local minima
        and maxima which easily hurdle an optimizer from finding the global minimum.

        In this setting, Q-learning performs better than policy gradient
        """
        input_param = ng.p.Dict(
            choice1=ng.p.Choice(["1", "2", "3"]),
            choice2=ng.p.Choice(["a", "b", "c"]),
        )

        def obj_func(sampled_sol: Dict[str, torch.Tensor]) -> torch.Tensor:
            # sampled_sol format:
            #    key = choice_name
            #    val = choice_idx (a tensor of length `batch_size`)
            assert list(sampled_sol.values())[0].dim() == 1
            batch_size = list(sampled_sol.values())[0].shape[0]
            result = torch.zeros(batch_size, 1)
            choice1 = sampled_sol["choice1"]
            choice2 = sampled_sol["choice2"]
            for i in range(batch_size):
                if choice1[i] == 1 and choice2[i] == 1:
                    result[i] = 0.4
                elif choice1[i] == 0 and choice2[i] == 0:
                    result[i] = 0.43
                elif choice1[i] == 1 or choice2[i] == 1:
                    result[i] = 0.9
                else:
                    result[i] = 0.45
            return result

        batch_size = 32
        n_generations = 100
        repeat = 10

        qlearning_res = self.run_q_learning_optimizer(
            input_param, obj_func, batch_size, n_generations, repeat
        )
        pg_res = self.run_policy_gradient_optimizer(
            input_param, obj_func, batch_size, n_generations, repeat
        )
        print(f"QLearning results over {repeat} repeats: {qlearning_res}")
        print(f"PG results over {repeat} repeats: {pg_res}")

        assert (
            np.mean(qlearning_res) < 0.42
        ), "QLearning should end up better than local minimum (0.43)"
        assert np.mean(qlearning_res) < np.mean(
            pg_res
        ), f"In this setting. qlearning should be better than policy gradient over {repeat} repeats"

    def test_sol_to_tensors(self):
        input_param = discrete_input_param()
        sampled_sol = {
            "choice1": torch.tensor([0, 1, 2]),
            "choice2": torch.tensor([1, 2, 0]),
            "choice3": torch.tensor([0, 1, 0]),
            "choice4": torch.tensor([4, 3, 2]),
            "choice5": torch.tensor([1, 2, 3]),
        }
        tensor = torch.FloatTensor(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
        sampled_tensor = sol_to_tensors(sampled_sol, input_param)
        self.assertTrue(torch.all(tensor == sampled_tensor))

    def test_bayesian_optimizer_its_random_mutation_discrete(self):
        acq_type = "its"
        mutation_type = "random"
        input_param = discrete_input_param()
        gt_net = create_ground_truth_net(input_param)
        obj_func = create_discrete_choice_obj_func(input_param, gt_net)
        optimizer = BayesianOptimizer(
            param=input_param,
            obj_func=obj_func,
            start_temp=1.0,
            min_temp=0.0,
            acq_type=acq_type,
            mutation_type=mutation_type,
        )
        sampled_solution = {
            "choice1": torch.tensor([0]),
            "choice2": torch.tensor([1]),
            "choice3": torch.tensor([0]),
            "choice4": torch.tensor([1]),
            "choice5": torch.tensor([0]),
        }
        optimizer._maintain_best_solutions(sampled_solution, torch.tensor([0.0]))
        # mutation
        np.random.choice = Mock(return_value=np.array(["choice1"]))
        torch.randint = Mock(return_value=torch.tensor([1]))
        mutated_solution = {
            "choice1": torch.tensor([1]),
            "choice2": torch.tensor([1]),
            "choice3": torch.tensor([0]),
            "choice4": torch.tensor([1]),
            "choice5": torch.tensor([0]),
        }
        sampled_sol = optimizer.sample(1, 1 / len(input_param))
        self.assertEqual(sampled_sol, mutated_solution)
        # acquisition
        predictor = [
            create_ground_truth_net(input_param),
            create_ground_truth_net(input_param),
        ]
        acq_reward = torch.tensor([-2.5])
        torch.normal = Mock(return_value=acq_reward)
        self.assertEqual(
            optimizer.acquisition("its", mutated_solution, predictor), acq_reward
        )
