#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional, Tuple, Union

import reagent.core.types as rlt
import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import RLParameters
from reagent.gym.policies import Policy
from reagent.gym.policies.samplers.discrete_sampler import (
    GreedyActionSampler,
    SoftmaxActionSampler,
)
from reagent.gym.policies.samplers.top_k_sampler import TopKSampler
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_serving_scorer,
    parametric_dqn_serving_scorer,
)
from reagent.gym.policies.scorers.slate_q_scorer import slate_q_serving_scorer
from reagent.models.actor import LOG_PROB_MIN, LOG_PROB_MAX


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorUnwrapper as ActorPredictorUnwrapper,
        FbDiscreteDqnPredictorUnwrapper as DiscreteDqnPredictorUnwrapper,
        FbParametricPredictorUnwrapper as ParametricDqnPredictorUnwrapper,
    )
else:
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
        rl_parameters = kwargs.get("rl_parameters", None)
        return DiscreteDQNPredictorPolicy(serving_module, rl_parameters)
    elif module_name.endswith("ActorPredictorWrapper"):
        return ActorPredictorPolicy(predictor=ActorPredictorUnwrapper(serving_module))
    elif module_name.endswith("ParametricDqnPredictorWrapper"):
        # TODO: remove this dependency
        max_num_actions = kwargs.get("max_num_actions", None)
        assert (
            max_num_actions is not None
        ), f"max_num_actions not given for Parametric DQN."
        q_network = ParametricDqnPredictorUnwrapper(serving_module)

        # TODO: write SlateQ Wrapper
        slate_size = kwargs.get("slate_size", None)
        if slate_size is not None:
            scorer = slate_q_serving_scorer(
                num_candidates=max_num_actions, q_network=q_network
            )
            sampler = TopKSampler(k=slate_size)
        else:
            sampler = GreedyActionSampler()
            scorer = parametric_dqn_serving_scorer(
                max_num_actions=max_num_actions, q_network=q_network
            )
        return Policy(scorer=scorer, sampler=sampler)
    else:
        raise NotImplementedError(
            f"Predictor policy for serving module {serving_module} not available."
        )


class DiscreteDQNPredictorPolicy(Policy):
    def __init__(self, wrapped_dqn_predictor, rl_parameters: Optional[RLParameters]):
        if rl_parameters and rl_parameters.softmax_policy:
            self.sampler = SoftmaxActionSampler(temperature=rl_parameters.temperature)
        else:
            self.sampler = GreedyActionSampler()
        self.scorer = discrete_dqn_serving_scorer(
            q_network=DiscreteDqnPredictorUnwrapper(wrapped_dqn_predictor)
        )

    @torch.no_grad()
    def act(
        self,
        obs: Union[rlt.ServingFeatureData, Tuple[torch.Tensor, torch.Tensor]],
        possible_actions_mask: Optional[torch.Tensor],
    ) -> rlt.ActorOutput:
        """Input is either state_with_presence, or
        ServingFeatureData (in the case of sparse features)"""
        assert isinstance(obs, tuple)
        if isinstance(obs, rlt.ServingFeatureData):
            state: rlt.ServingFeatureData = obs
        else:
            state = rlt.ServingFeatureData(
                float_features_with_presence=obs,
                id_list_features={},
                id_score_list_features={},
            )
        scores = self.scorer(state, possible_actions_mask)
        return self.sampler.sample_action(scores).cpu().detach()


class ActorPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    @torch.no_grad()
    def act(
        self,
        obs: Union[rlt.ServingFeatureData, Tuple[torch.Tensor, torch.Tensor]],
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> rlt.ActorOutput:
        """Input is either state_with_presence, or
        ServingFeatureData (in the case of sparse features)"""
        assert isinstance(obs, tuple)
        if isinstance(obs, rlt.ServingFeatureData):
            state: rlt.ServingFeatureData = obs
        else:
            state = rlt.ServingFeatureData(
                float_features_with_presence=obs,
                id_list_features={},
                id_score_list_features={},
            )
        output = self.predictor(*state)
        if isinstance(output, tuple):
            action, log_prob = output
            log_prob = log_prob.clamp(LOG_PROB_MIN, LOG_PROB_MAX)
            return rlt.ActorOutput(action=action.cpu(), log_prob=log_prob.cpu())
        else:
            return rlt.ActorOutput(action=output.cpu())
