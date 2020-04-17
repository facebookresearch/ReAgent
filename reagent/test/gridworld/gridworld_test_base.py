#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import tempfile

import torch
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.test.gridworld.gridworld_base import DISCOUNT, GridworldBase
from reagent.test.gridworld.gridworld_evaluator import GridworldEvaluator
from reagent.torch_utils import export_module_to_buffer


logger = logging.getLogger(__name__)


class GridworldTestBase(HorizonTestBase):
    def setUp(self):
        self.check_tolerance = True
        self.test_save_load = True
        self.num_epochs = 5
        self.tolerance_threshold = (
            GridworldEvaluator.ABS_ERR_THRES * GridworldBase.REWARD_SCALE
        )
        self.run_pre_training_eval = True
        super().setUp()

    def evaluate_gridworld(
        self, environment, evaluator, trainer, use_gpu, one_hot_action=True
    ):
        pre_training_loss = None
        if self.run_pre_training_eval:
            predictor = self.get_predictor(trainer, environment)

            evaluator.evaluate(predictor)
            print("Pre-Training eval: ", evaluator.mc_loss[-1])
            pre_training_loss = evaluator.mc_loss[-1]

        for _ in range(self.num_epochs):
            samples = environment.generate_samples(10240, 1.0, DISCOUNT)

            if trainer.rl_parameters.reward_boost is not None:
                # Reverse any reward boost
                rewards_update = []
                for action, reward in zip(samples.actions, samples.rewards):
                    rewards_update.append(
                        reward - trainer.rl_parameters.reward_boost.get(action, 0.0)
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

        predictor = self.get_predictor(trainer, environment)
        evaluator.evaluate(predictor)
        print("Post-Training eval: ", evaluator.mc_loss[-1])
        if self.check_tolerance:
            self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)
        if self.run_pre_training_eval:
            self.assertGreater(pre_training_loss, evaluator.mc_loss[-1])

        if self.test_save_load:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = os.path.join(tmpdirname, "model")

                serving_module_bytes = export_module_to_buffer(
                    predictor.model
                ).getvalue()
                logger.info("Saving TorchScript predictor to {}".format(tmp_path))
                with open(tmp_path, "wb") as output_fp:
                    output_fp.write(serving_module_bytes)
                new_predictor = type(predictor)(torch.jit.load(tmp_path))
                evaluator.evaluate(new_predictor)
                print("Post-ONNX eval: ", evaluator.mc_loss[-1])
                if self.check_tolerance:
                    self.assertLess(evaluator.mc_loss[-1], self.tolerance_threshold)
