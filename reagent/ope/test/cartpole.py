#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging

import gym
import torch
from reagent.ope.estimators.sequential_estimators import (
    Action,
    ActionDistribution,
    ActionSpace,
    IPSEstimator,
    Model,
    NeuralDualDICE,
    RandomRLPolicy,
    RewardProbability,
    RLEstimatorInput,
    RLPolicy,
    State,
    StateDistribution,
    Transition,
)
from reagent.ope.utils import RunningAverage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EPISODES = 200
MAX_HORIZON = 250
GAMMA = 0.99


class ComboPolicy(RLPolicy):
    # Weighted combination between two given policies
    def __init__(self, action_space: ActionSpace, weights, policies):
        assert len(weights) == len(policies)
        self._weights = weights
        self._policies = policies
        self._action_space = action_space

    def action_dist(self, state: State) -> ActionDistribution:
        weighted_policies = [
            w * p(state).values for w, p in zip(self._weights, self._policies)
        ]
        weighted = torch.stack(weighted_policies).sum(0)
        return self._action_space.distribution(weighted)


class PyTorchPolicy(RLPolicy):
    def __init__(self, action_space: ActionSpace, model):
        self._action_space = action_space
        self._model = model
        self._softmax = torch.nn.Softmax(dim=0)

    def action_dist(self, state: State) -> ActionDistribution:
        self._model.eval()
        dist = self._model(torch.tensor(state.value, dtype=torch.float).reshape(1, -1))[
            0
        ]
        return self._action_space.distribution(self._softmax(dist))


class EnvironmentModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layers, activation):
        super(EnvironmentModel.Network, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._hidden_layers = hidden_layers
        self._activation = activation

        self.layers = []
        dim = self._state_dim + self._action_dim
        for _ in range(self._hidden_layers):
            self.layers.append(torch.nn.Linear(dim, self._hidden_dim))
            self.layers.append(self._activation())
            dim = self._hidden_dim
        # Output is the next state and its reward
        self.layers.append(torch.nn.Linear(dim, self._state_dim + 1))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat((state, action), dim=1)
        return self.model(x)


class ModelWrapper(Model):
    def __init__(self, model: EnvironmentModel, device=None):
        self._model = model
        self._device = device
        self._model.to(self._device)

    def next_state_reward_dist(self, state: State, action: Action) -> StateDistribution:
        self._model.eval()
        state_reward_tensor = (
            self._model(
                torch.tensor(state.value, dtype=torch.float)
                .reshape(-1, self._model._state_dim)
                .to(self._device),
                torch.nn.functional.one_hot(
                    torch.tensor(action.value, dtype=torch.long),
                    self._model._action_dim,
                )
                .reshape(-1, self._model._action_dim)
                .float()
                .to(self._device),
            )
            .reshape(-1)
            .cpu()
        )
        return {
            State(state_reward_tensor[: self._model._state_dim]): RewardProbability(
                state_reward_tensor[-1].item()
            )
        }

    def to(self, device):
        self._model.to(device)


def generate_logs(episodes: int, max_horizon: int, policy: RLPolicy):
    """
    Args:
        episodes: number of episodes to generate
        max_horizon: max horizon of each episode
        policy: RLPolicy which uses real-valued states
    """
    log = []
    env = gym.make("CartPole-v0")
    for _ in range(episodes):
        init_state = env.reset()
        cur_state = init_state
        mdp = []
        for _ in range(max_horizon):
            action_dist = policy(State(cur_state))
            action = action_dist.sample()[0].value
            action_prob = action_dist.probability(Action(action))
            next_state, _, done, _ = env.step(action)
            mdp.append(
                Transition(
                    last_state=State(cur_state),
                    action=Action(action),
                    action_prob=action_prob,
                    state=State(next_state),
                    reward=1.0,
                    status=Transition.Status.NORMAL,
                )
            )
            cur_state = next_state
            if done:
                log.append(mdp)
                break
        log.append(mdp)
    return log


def zeta_nu_loss_callback(losses, estimated_values, input: RLEstimatorInput):
    def callback_fn(zeta_loss, nu_loss, estimator):
        losses.append((zeta_loss, nu_loss))
        estimated_values.append(estimator._compute_estimates(input))

    return callback_fn


def estimate_value(episodes: int, max_horizon: int, policy: RLPolicy, gamma: float):
    avg = RunningAverage()
    env = gym.make("CartPole-v0")
    for _ in range(episodes):
        init_state = env.reset()
        cur_state = init_state
        r = 0.0
        discount = 1.0
        for _ in range(max_horizon):
            action_dist = policy(State(cur_state))
            action = action_dist.sample()[0].value
            next_state, _, done, _ = env.step(action)
            reward = 1.0
            r += reward * discount
            discount *= gamma
            if done:
                break
            cur_state = next_state
        avg.add(r)
    return avg.average


def run_dualdice_test(model_path: str, alpha: float):
    device = torch.device("cuda") if torch.cuda.is_available() else None
    logger.info(f"Device - {device}")
    model = torch.jit.load(model_path)
    model = model.dqn_with_preprocessor.model

    random_policy = RandomRLPolicy(ActionSpace(2))
    model_policy = PyTorchPolicy(ActionSpace(2), model)
    target_policy = ComboPolicy(
        ActionSpace(2), [0.7, 0.3], [model_policy, random_policy]
    )
    behavior_policy = ComboPolicy(
        ActionSpace(2),
        [0.55 + 0.15 * alpha, 0.45 - 0.15 * alpha],
        [model_policy, random_policy],
    )

    ground_truth = estimate_value(NUM_EPISODES, MAX_HORIZON, target_policy, GAMMA)
    log_policy_value = estimate_value(NUM_EPISODES, MAX_HORIZON, behavior_policy, GAMMA)
    trained_policy_value = estimate_value(
        NUM_EPISODES, MAX_HORIZON, model_policy, GAMMA
    )

    logger.info(f"Target Policy Ground Truth value: {ground_truth}")
    logger.info(f"Behavior Policy Ground Truth value: {log_policy_value}")
    logger.info(f"Model Policy Ground Truth value: {trained_policy_value}")

    log = generate_logs(NUM_EPISODES, MAX_HORIZON, behavior_policy)

    inp = RLEstimatorInput(
        gamma=GAMMA, log=log, target_policy=target_policy, discrete_states=False
    )
    ips = IPSEstimator()
    dualdice_losses = []
    dualdice_values = []
    dualdice = NeuralDualDICE(
        state_dim=4,
        action_dim=2,
        deterministic_env=True,
        average_next_v=False,
        value_lr=0.003,
        zeta_lr=0.003,
        batch_size=2048,
        reporting_frequency=1000,
        training_samples=100000,
        loss_callback_fn=zeta_nu_loss_callback(dualdice_losses, dualdice_values, inp),
        device=device,
    )

    ips_result = ips.evaluate(inp)
    dd_result = dualdice.evaluate(inp)

    return {
        "ips_estimate": ips_result,
        "dualdice_estimate": dd_result,
        "ground_truth": ground_truth,
        "dualdice_losses": dualdice_losses,
        "dualdice_estimates_per_epoch": dualdice_values,
    }


if __name__ == "__main__":
    run_dualdice_test(
        "/mnt/vol/gfsfblearner-nebraska/flow/data/2020-07-27/a56cd422-794b-4866-9b73-5de95fb65700/207851498_207851498_0.pt",
        0.0,
    )
