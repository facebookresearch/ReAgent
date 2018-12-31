#!/usr/bin/env python3

import os
import tempfile
import unittest

from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.test.gridworld.gridworld_base import DISCOUNT
from ml.rl.training.ddpg_trainer import DDPGTrainer


class GridworldTestBase(unittest.TestCase):
    def setUp(self):
        self.check_tolerance = True
        self.test_save_load = True
        self.num_epochs = 5
        self.tolerance_threshold = 0.1
        self.run_pre_training_eval = True
        SummaryWriterContext._reset_globals()

    def tearDown(self):
        SummaryWriterContext._reset_globals()

    def evaluate_gridworld(
        self, environment, evaluator, trainer, exporter, use_gpu, one_hot_action=True
    ):
        if self.run_pre_training_eval:
            predictor = exporter.export()

            evaluator.evaluate(predictor)
            print("Pre-Training eval: ", evaluator.mc_loss[-1])
            self.assertGreater(evaluator.mc_loss[-1], 0.09)

        for _ in range(self.num_epochs):
            samples = environment.generate_samples(10240, 1.0, DISCOUNT)

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
                samples = samples._replace(rewards=rewards_update)

            tdps = environment.preprocess_samples(
                samples,
                self.minibatch_size,
                use_gpu=use_gpu,
                one_hot_action=one_hot_action,
            )

            for tdp in tdps:
                trainer.train(tdp)

        # Test actor if it exists
        if isinstance(trainer, DDPGTrainer):
            # Make sure actor predictor works
            actor = trainer.predictor(actor=True)
            # Make sure all actions are optimal
            error = evaluator.evaluate_actor(actor, thres=0.2)
            print("gridworld optimal action match MAE: {0:.3f}".format(error))

        predictor = exporter.export()
        predictorClass = predictor.__class__
        evaluator.evaluate(predictor)
        print("Post-Training eval: ", evaluator.mc_loss[-1])
        if self.check_tolerance:
            self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)

        if self.test_save_load:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = os.path.join(tmpdirname, "model")
                predictor.save(tmp_path, "minidb")
                new_predictor = predictorClass.load(tmp_path, "minidb", False)
                evaluator.evaluate(new_predictor)
                print("Post-ONNX eval: ", evaluator.mc_loss[-1])
                self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)
