#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch


class SoftUpdate(torch.optim.Optimizer):
    def __init__(self, target_params, source_params, tau=0.1):
        """
        Perform soft-update on target_params. Soft-update gradually blends
        source_params into target_params with this update equation:

            target_param = tau * source_param + (1 - tau) * target_param
        """
        target_params = list(target_params)
        source_params = list(source_params)

        if len(target_params) != len(source_params):
            raise ValueError(
                "target and source must have the same number of parameters"
            )

        for t_param, s_param in zip(target_params, source_params):
            if t_param.shape != s_param.shape:
                raise ValueError(
                    "The shape of target parameter doesn't match that of the source"
                )

        params = target_params + source_params
        defaults = dict(
            tau=tau, lr=1.0
        )  # set a dummy learning rate because optimizers are expected to have one
        super().__init__(params, defaults)

        for group in self.param_groups:
            tau = group["tau"]
            if tau > 1.0 or tau < 0.0:
                raise ValueError(f"tau should be in [0.0, 1.0]; got {tau}")

    @classmethod
    def make_optimizer_scheduler(cls, target_params, source_params, tau):
        su = cls(target_params, source_params, tau)
        return {"optimizer": su}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            n = len(params)
            tau = group["tau"]
            for target_param, source_param in zip(params[: n // 2], params[n // 2 :]):
                if target_param is source_param:
                    # skip soft-updating when the target network share s the parameter with
                    # the network being train.
                    continue
                new_param = tau * source_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(new_param)
        return loss
