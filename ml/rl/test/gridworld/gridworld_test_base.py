#!/usr/bin/env python3

import os
import tempfile
import unittest

from ml.rl.test.gridworld.gridworld_base import DISCOUNT


class GridworldTestBase(unittest.TestCase):
    def setUp(self):
        self.check_tolerance = True
        self.test_save_load = True
        self.num_epochs = 5
        self.tolerance_threshold = 0.1
        self.run_pre_training_eval = True

    def evaluate_gridworld(
        self, environment, evaluator, trainer, exporter, use_gpu, one_hot_action=True
    ):
        if self.run_pre_training_eval:
            predictor = exporter.export()

            evaluator.evaluate(predictor)
            print(
                "Pre-Training eval: ",
                evaluator.mc_loss[-1],
                evaluator.reward_doubly_robust[-1]
                if len(evaluator.reward_doubly_robust) > 0
                else "None",
            )
            self.assertGreater(evaluator.mc_loss[-1], 0.09)

        for _ in range(self.num_epochs):
            samples = environment.generate_samples(10240, 1.0, DISCOUNT)
            true_values = environment.true_values_for_sample(
                samples.states, samples.actions, False
            )
            # Hijack the reward timeline to insert the ground truth
            samples.episode_values = true_values

            if (
                hasattr(trainer.parameters.rl, "reward_boost")
                and trainer.parameters.rl.reward_boost is not None
            ):
                # Reverse any reward boost
                rewards_update = []
                for action, reward in zip(samples.actions, samples.rewards):
                    rewards_update.append(
                        reward - trainer.parameters.rl.reward_boost.get(action, 0.0)
                    )
                samples.rewards = rewards_update

            tdps = environment.preprocess_samples(
                samples,
                self.minibatch_size,
                use_gpu=use_gpu,
                one_hot_action=one_hot_action,
            )

            for tdp in tdps:
                trainer.train(tdp)

        predictor = exporter.export()
        predictorClass = predictor.__class__
        evaluator.evaluate(predictor)
        print(
            "Post-Training eval: ",
            evaluator.mc_loss[-1],
            evaluator.reward_doubly_robust[-1]
            if len(evaluator.reward_doubly_robust) > 0
            else "None",
        )
        if self.check_tolerance:
            self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)

        if self.test_save_load:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = os.path.join(tmpdirname, "model")
                predictor.save(tmp_path, "minidb")
                new_predictor = predictorClass.load(tmp_path, "minidb", False)
                evaluator.evaluate(new_predictor)
                print(
                    "Post-ONNX eval: ",
                    evaluator.mc_loss[-1],
                    evaluator.reward_doubly_robust[-1]
                    if len(evaluator.reward_doubly_robust) > 0
                    else "None",
                )
                self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)
