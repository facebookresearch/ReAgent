#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import gym
import numpy as np
import reagent.types as rlt
from reagent.core.dataclasses import dataclass
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.envs.wrappers.recsim import ValueWrapper
from reagent.gym.preprocessors.default_preprocessors import RecsimObsPreprocessor
from recsim import choice_model, utils
from recsim.environments import interest_evolution, interest_exploration
from recsim.simulator import environment, recsim_gym


logger = logging.getLogger(__name__)


def dot_value_fn(user, doc):
    return np.inner(user, doc)


def multi_selection_value_fn(user, doc):
    return (np.inner(user, doc) + 1.0) / 2.0


@dataclass
class RecSim(EnvWrapper):
    num_candidates: int
    slate_size: int
    resample_documents: bool = True
    single_selection: bool = True
    is_interest_exploration: bool = False
    initial_seed: int = 1

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        if self.is_interest_exploration and not self.single_selection:
            raise NotImplementedError(
                "Multiselect interest exploration not implemented"
            )

    def make(self) -> gym.Env:
        env_config = {
            "slate_size": self.slate_size,
            "seed": self.initial_seed,
            "num_candidates": self.num_candidates,
            "resample_documents": self.resample_documents,
        }
        if self.is_interest_exploration:
            env = interest_exploration.create_environment(env_config)
            return ValueWrapper(env, lambda user, doc: 0.0)

        if self.single_selection:
            env = interest_evolution.create_environment(env_config)
            return ValueWrapper(env, dot_value_fn)
        else:
            env = create_multiclick_environment(env_config)
            return ValueWrapper(env, multi_selection_value_fn)

    def make(self) -> gym.Env:
        env_config = {
            "slate_size": self.slate_size,
            "seed": 1,
            "num_candidates": self.num_candidates,
            "resample_documents": self.resample_documents,
        }
        if self.is_interest_exploration:
            env = interest_exploration.create_environment(env_config)
            return ValueWrapper(env, lambda user, doc: 0.0)

        if self.single_selection:
            env = interest_evolution.create_environment(env_config)
            return ValueWrapper(env, dot_value_fn)
        else:
            env = create_multiclick_environment(env_config)
            return ValueWrapper(env, multi_selection_value_fn)

    def obs_preprocessor(self, obs: np.ndarray) -> rlt.FeatureData:
        # TODO: remove RecsimObsPreprocessor and move it here
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        return preprocessor(obs)

    def serving_obs_preprocessor(self, obs: np.ndarray):
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        return preprocessor(obs)


class MulticlickIEvUserModel(interest_evolution.IEvUserModel):
    def simulate_response(self, documents):
        responses = [self._response_model_ctor() for _ in documents]
        self.choice_model.score_documents(
            self._user_state, [doc.create_observation() for doc in documents]
        )
        selected_indices = self.choice_model.choose_items()
        for i, response in enumerate(responses):
            response.quality = documents[i].quality
            response.cluster_id = documents[i].cluster_id
        for selected_index in selected_indices:
            self._generate_click_response(
                documents[selected_index], responses[selected_index]
            )
        return responses


class UserState(interest_evolution.IEvUserState):
    def score_document(self, doc_obs):
        scores = super().score_document(doc_obs)
        # return choice_model.softmax(scores)
        return (scores + 1) / 2


def create_multiclick_environment(env_config):
    """Creates an interest evolution environment."""

    def choice_model_ctor(*args, **kwargs):
        return choice_model.DependentClickModel(
            next_probs=[0.8 ** (i + 1) for i in range(env_config["slate_size"])],
            slate_size=env_config["slate_size"],
            score_scaling=1.0,
        )

    user_model = MulticlickIEvUserModel(
        env_config["slate_size"],
        choice_model_ctor=choice_model_ctor,
        response_model_ctor=interest_evolution.IEvResponse,
        user_state_ctor=UserState,
        seed=env_config["seed"],
    )

    document_sampler = interest_evolution.UtilityModelVideoSampler(
        doc_ctor=interest_evolution.IEvVideo, seed=env_config["seed"]
    )

    ievenv = environment.Environment(
        user_model,
        document_sampler,
        env_config["num_candidates"],
        env_config["slate_size"],
        resample_documents=env_config["resample_documents"],
    )

    return recsim_gym.RecSimGymEnv(
        ievenv,
        interest_evolution.clicked_watchtime_reward,
        utils.aggregate_video_cluster_metrics,
        utils.write_video_cluster_metrics,
    )
