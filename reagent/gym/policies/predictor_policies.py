#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
from reagent.gym.policies import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_serving_scorer,
    parametric_dqn_serving_scorer,
)


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorUnwrapper as ActorPredictorUnwrapper,
        FbDiscreteDqnPredictorUnwrapper as DiscreteDqnPredictorUnwrapper,
        FbParametricPredictorUnwrapper as ParametricDqnPredictorUnwrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import (
        ActorPredictorUnwrapper,
        DiscreteDqnPredictorUnwrapper,
        ParametricDqnPredictorUnwrapper,
    )


def create_predictor_policy_from_model(serving_module, **kwargs) -> Policy:
    """
    serving_module is the result of ModelManager.build_serving_module().
    This function creates a Policy for gym environments.
    """
    module_name = serving_module.original_name
    if module_name.endswith("DiscreteDqnPredictorWrapper"):
        expand_args = kwargs.get("expand_args", False)
        scorer = discrete_dqn_serving_scorer(
            q_network=DiscreteDqnPredictorUnwrapper(serving_module)
        )
        return Policy(
            scorer=scorer, sampler=GreedyActionSampler(), expand_args=expand_args
        )
    elif module_name.endswith("ActorPredictorWrapper"):
        return ActorPredictorPolicy(predictor=ActorPredictorUnwrapper(serving_module))
    elif module_name.endswith("ParametricDqnPredictorWrapper"):
        # TODO: remove this dependency
        max_num_actions = kwargs.get("max_num_actions", None)
        assert (
            max_num_actions is not None
        ), f"max_num_actions not given for Parametric DQN."
        sampler = GreedyActionSampler()
        scorer = parametric_dqn_serving_scorer(
            max_num_actions=max_num_actions,
            q_network=ParametricDqnPredictorUnwrapper(serving_module),
        )
        return Policy(scorer=scorer, sampler=sampler)
    else:
        raise NotImplementedError(
            f"Predictor policy for serving module {serving_module} not available."
        )


class ActorPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    @torch.no_grad()
    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        action = self.predictor(obs).cpu()
        # TODO: return log_probs as well
        return rlt.ActorOutput(action=action)
