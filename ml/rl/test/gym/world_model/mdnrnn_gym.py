#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Learn a world model on gym environments
"""
import argparse
import json
import logging
import sys
from typing import Dict, Optional

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.evaluation.world_model_evaluator import (
    FeatureImportanceEvaluator,
    FeatureSensitivityEvaluator,
)
from ml.rl.models.mdn_rnn import MDNRNNMemoryPool
from ml.rl.models.world_model import MemoryNetwork
from ml.rl.test.gym.open_ai_gym_environment import (
    EnvType,
    ModelType,
    OpenAIGymEnvironment,
)
from ml.rl.test.gym.run_gym import dict_to_np, get_possible_actions
from ml.rl.thrift.core.ttypes import MDNRNNParameters
from ml.rl.training.rl_dataset import RLDataset
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)

USE_CPU = -1


def loss_to_num(losses):
    return {k: v.item() for k, v in losses.items()}


def multi_step_sample_generator(
    gym_env: OpenAIGymEnvironment,
    num_transitions: int,
    max_steps: Optional[int],
    multi_steps: int,
):
    """
    Convert gym env multi-step sample format to mdn-rnn multi-step sample format
    """
    samples = gym_env.generate_random_samples(
        num_transitions=num_transitions,
        use_continuous_action=True,
        max_step=max_steps,
        multi_steps=multi_steps,
    )

    transition_terminal_index = [-1]
    for i in range(1, len(samples.mdp_ids)):
        if samples.terminals[i][0] is True:
            assert len(samples.terminals[i]) == 1
            transition_terminal_index.append(i)

    for i in range(len(transition_terminal_index) - 1):
        episode_start = transition_terminal_index[i] + 1
        episode_end = transition_terminal_index[i + 1]

        for j in range(episode_start, episode_end + 1):
            if len(samples.terminals[j]) != multi_steps:
                continue
            state = dict_to_np(
                samples.states[j], np_size=gym_env.state_dim, key_offset=0
            )
            action = dict_to_np(
                samples.actions[j],
                np_size=gym_env.action_dim,
                key_offset=gym_env.state_dim,
            )
            next_actions = np.float32(
                [
                    dict_to_np(
                        samples.next_actions[j][k],
                        np_size=gym_env.action_dim,
                        key_offset=gym_env.state_dim,
                    )
                    for k in range(multi_steps)
                ]
            )
            next_states = np.float32(
                [
                    dict_to_np(
                        samples.next_states[j][k],
                        np_size=gym_env.state_dim,
                        key_offset=0,
                    )
                    for k in range(multi_steps)
                ]
            )
            rewards = np.float32(samples.rewards[j])
            terminals = np.float32(samples.terminals[j])
            not_terminals = np.logical_not(terminals)
            mdnrnn_state = np.vstack((state, next_states))[:-1]
            mdnrnn_action = np.vstack((action, next_actions))[:-1]

            assert mdnrnn_state.shape == (multi_steps, gym_env.state_dim)
            assert mdnrnn_action.shape == (multi_steps, gym_env.action_dim)
            assert rewards.shape == (multi_steps,)
            assert next_states.shape == (multi_steps, gym_env.state_dim)
            assert next_actions.shape == (multi_steps, gym_env.action_dim)
            assert not_terminals.shape == (multi_steps,)

            yield mdnrnn_state, mdnrnn_action, rewards, next_states, next_actions, not_terminals


def get_replay_buffer(
    num_episodes: int,
    seq_len: int,
    max_step: Optional[int],
    gym_env: OpenAIGymEnvironment,
):
    num_transitions = num_episodes * max_step
    replay_buffer = MDNRNNMemoryPool(max_replay_memory_size=num_transitions)
    for (
        mdnrnn_state,
        mdnrnn_action,
        rewards,
        next_states,
        _,
        not_terminals,
    ) in multi_step_sample_generator(
        gym_env,
        num_transitions=num_transitions,
        max_steps=max_step,
        multi_steps=seq_len,
    ):
        replay_buffer.insert_into_memory(
            mdnrnn_state, mdnrnn_action, next_states, rewards, not_terminals
        )

    return replay_buffer


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a Mixture-Density-Network RNN net to learn an OpenAI"
        " Gym environment, i.e., predict next state, reward, and"
        " terminal signal using current state and action"
    )
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="If set, will use GPU with specified ID. Otherwise will use CPU.",
        default=USE_CPU,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="If set, use logging level specified (debug, info, warning, error, "
        "critical). Else defaults to info.",
        default="info",
    )
    parser.add_argument(
        "-f",
        "--feature_importance",
        action="store_true",
        help="If set, feature importance will be calculated after the training",
    )
    parser.add_argument(
        "-s",
        "--feature_sensitivity",
        action="store_true",
        help="If set, state feature sensitivity by varying actions will be"
        " calculated after the training",
    )
    parser.add_argument(
        "-e",
        "--save_embedding_to_path",
        help="If a file path is provided, save a RLDataset with states embedded"
        " by the trained world model",
    )
    args = parser.parse_args(args)

    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception("Logging level {} not valid level.".format(args.log_level))
    else:
        logger.setLevel(getattr(logging, args.log_level.upper()))

    with open(args.parameters, "r") as f:
        params = json.load(f)

    use_gpu = args.gpu_id != USE_CPU
    mdnrnn_gym(
        params,
        use_gpu,
        args.feature_importance,
        args.feature_sensitivity,
        args.save_embedding_to_path,
    )


def mdnrnn_gym(
    params: Dict,
    use_gpu: bool,
    feature_importance: bool = False,
    feature_sensitivity: bool = False,
    save_embedding_to_path: Optional[str] = None,
):
    logger.info("Running gym with params")
    logger.info(params)

    env_type = params["env"]
    env = OpenAIGymEnvironment(env_type, epsilon=1.0, softmax_policy=True, gamma=0.99)

    trainer = create_trainer(params, env, use_gpu)
    _, _, trainer = train_sgd(
        env,
        trainer,
        use_gpu,
        "{} test run".format(env_type),
        params["mdnrnn"]["minibatch_size"],
        **params["run_details"],
    )
    feature_importance_map, feature_sensitivity_map, dataset = None, None, None
    if feature_importance:
        feature_importance_map = calculate_feature_importance(
            env, trainer, use_gpu, **params["run_details"]
        )
    if feature_sensitivity:
        feature_sensitivity_map = calculate_feature_sensitivity_by_actions(
            env, trainer, use_gpu, **params["run_details"]
        )
    if save_embedding_to_path:
        dataset = RLDataset(save_embedding_to_path)
        create_embed_rl_dataset(env, trainer, dataset, use_gpu, **params["run_details"])
        dataset.save()
    return env, trainer, feature_importance_map, feature_sensitivity_map, dataset


def calculate_feature_importance(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    seq_len: int = 5,
    num_test_episodes: int = 100,
    max_steps: Optional[int] = None,
    **kwargs,
):
    feature_importance_evaluator = FeatureImportanceEvaluator(
        trainer,
        discrete_action=gym_env.action_type == EnvType.DISCRETE_ACTION,
        state_feature_num=gym_env.state_dim,
        action_feature_num=gym_env.action_dim,
        sorted_action_feature_start_indices=list(range(gym_env.action_dim)),
        sorted_state_feature_start_indices=list(range(gym_env.state_dim)),
    )
    test_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    test_batch = test_replay_buffer.sample_memories(
        test_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
    )
    feature_loss_vector = feature_importance_evaluator.evaluate(test_batch)[
        "feature_loss_increase"
    ]
    feature_importance_map = {}
    for i in range(gym_env.action_dim):
        print(
            "action {}, feature importance: {}".format(i, feature_loss_vector[i].item())
        )
        feature_importance_map["action" + str(i)] = feature_loss_vector[i].item()
    for i in range(gym_env.state_dim):
        print(
            "state {}, feature importance: {}".format(
                i, feature_loss_vector[i + gym_env.action_dim].item()
            )
        )
        feature_importance_map["state" + str(i)] = feature_loss_vector[
            i + gym_env.action_dim
        ].item()
    return feature_importance_map


def create_embed_rl_dataset(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    dataset: RLDataset,
    use_gpu: bool = False,
    seq_len: int = 5,
    num_state_embed_episodes: int = 100,
    max_steps: Optional[int] = None,
    **kwargs,
):
    old_mdnrnn_mode = trainer.mdnrnn.mdnrnn.training
    trainer.mdnrnn.mdnrnn.eval()
    num_transitions = num_state_embed_episodes * max_steps
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # batch-compute state embedding
    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        next_action_batch,
        not_terminal_batch,
    ) = map(
        list,
        zip(
            *multi_step_sample_generator(
                gym_env=gym_env,
                num_transitions=num_transitions,
                max_steps=max_steps,
                multi_steps=seq_len,
            )
        ),
    )

    def concat_batch(batch):
        return torch.cat(
            [
                torch.tensor(
                    np.expand_dims(x, axis=1), dtype=torch.float, device=device
                )
                for x in batch
            ],
            dim=1,
        )

    # shape: seq_len x batch_size x feature_dim
    mdnrnn_state = concat_batch(state_batch)
    next_mdnrnn_state = concat_batch(next_state_batch)
    mdnrnn_action = concat_batch(action_batch)
    next_mdnrnn_action = concat_batch(next_action_batch)

    mdnrnn_input = rlt.StateAction(
        state=rlt.FeatureVector(float_features=mdnrnn_state),
        action=rlt.FeatureVector(float_features=mdnrnn_action),
    )
    next_mdnrnn_input = rlt.StateAction(
        state=rlt.FeatureVector(float_features=next_mdnrnn_state),
        action=rlt.FeatureVector(float_features=next_mdnrnn_action),
    )
    mdnrnn_output = trainer.mdnrnn(mdnrnn_input)
    next_mdnrnn_output = trainer.mdnrnn(next_mdnrnn_input)

    for i in range(len(state_batch)):
        # Embed the state as the hidden layer's output
        # until the previous step + current state
        hidden_embed = mdnrnn_output.all_steps_lstm_hidden[-2, i, :].squeeze()
        state_embed = np.hstack((hidden_embed.detach().numpy(), state_batch[i][-1]))
        next_hidden_embed = next_mdnrnn_output.all_steps_lstm_hidden[-2, i, :].squeeze()
        next_state_embed = np.hstack(
            (next_hidden_embed.detach().numpy(), next_state_batch[i][-1])
        )
        terminal = 1 - not_terminal_batch[i][-1]
        possible_actions, possible_actions_mask = get_possible_actions(
            gym_env, ModelType.PYTORCH_PARAMETRIC_DQN.value, False
        )
        possible_next_actions, possible_next_actions_mask = get_possible_actions(
            gym_env, ModelType.PYTORCH_PARAMETRIC_DQN.value, terminal
        )
        dataset.insert(
            state=state_embed,
            action=action_batch[i][-1],
            reward=reward_batch[i][-1],
            next_state=next_state_embed,
            next_action=next_action_batch[i][-1],
            terminal=terminal,
            possible_next_actions=possible_next_actions,
            possible_next_actions_mask=possible_next_actions_mask,
            time_diff=1,
            possible_actions=possible_actions,
            possible_actions_mask=possible_actions_mask,
            policy_id=0,
        )
    logger.info(
        "Insert {} transitions into a state embed dataset".format(len(state_batch))
    )
    trainer.mdnrnn.mdnrnn.train(old_mdnrnn_mode)
    return dataset


def calculate_feature_sensitivity_by_actions(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    seq_len: int = 5,
    num_test_episodes: int = 100,
    max_steps: Optional[int] = None,
    **kwargs,
):
    feature_sensitivity_evaluator = FeatureSensitivityEvaluator(
        trainer,
        state_feature_num=gym_env.state_dim,
        sorted_state_feature_start_indices=list(range(gym_env.state_dim)),
    )
    test_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    test_batch = test_replay_buffer.sample_memories(
        test_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
    )
    feature_sensitivity_vector = feature_sensitivity_evaluator.evaluate(test_batch)[
        "feature_sensitivity"
    ]
    feature_sensitivity_map = {}
    for i in range(gym_env.state_dim):
        feature_sensitivity_map["state" + str(i)] = feature_sensitivity_vector[i].item()
        print(
            "state {}, feature sensitivity: {}".format(
                i, feature_sensitivity_vector[i].item()
            )
        )
    return feature_sensitivity_map


def train_sgd(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    test_run_name: str,
    minibatch_size: int,
    seq_len: int = 5,
    num_train_episodes: int = 300,
    num_test_episodes: int = 100,
    max_steps: Optional[int] = None,
    train_epochs: int = 100,
    early_stopping_patience: int = 3,
    **kwargs,
):
    train_replay_buffer = get_replay_buffer(
        num_train_episodes, seq_len, max_steps, gym_env
    )
    valid_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    test_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    valid_loss_history = []

    num_batch_per_epoch = train_replay_buffer.memory_size // minibatch_size
    logger.info(
        "Collected data {} transitions.\n"
        "Training will take {} epochs, with each epoch having {} mini-batches"
        " and each mini-batch having {} samples".format(
            train_replay_buffer.memory_size,
            train_epochs,
            num_batch_per_epoch,
            minibatch_size,
        )
    )

    for i_epoch in range(train_epochs):
        for i_batch in range(num_batch_per_epoch):
            training_batch = train_replay_buffer.sample_memories(
                minibatch_size, use_gpu=use_gpu, batch_first=True
            )
            losses = trainer.train(training_batch, batch_first=True)
            logger.info(
                "{}-th epoch, {}-th minibatch: \n"
                "loss={}, bce={}, gmm={}, mse={} \n"
                "cum loss={}, cum bce={}, cum gmm={}, cum mse={}\n".format(
                    i_epoch,
                    i_batch,
                    losses["loss"],
                    losses["bce"],
                    losses["gmm"],
                    losses["mse"],
                    np.mean(trainer.cum_loss),
                    np.mean(trainer.cum_bce),
                    np.mean(trainer.cum_gmm),
                    np.mean(trainer.cum_mse),
                )
            )
        # earlystopping
        trainer.mdnrnn.mdnrnn.eval()
        valid_batch = valid_replay_buffer.sample_memories(
            valid_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
        )
        valid_losses = trainer.get_loss(
            valid_batch, state_dim=gym_env.state_dim, batch_first=True
        )
        valid_losses = loss_to_num(valid_losses)
        valid_loss_history.append(valid_losses)
        trainer.mdnrnn.mdnrnn.train()
        logger.info(
            "{}-th epoch, validate loss={}, bce={}, gmm={}, mse={}".format(
                i_epoch,
                valid_losses["loss"],
                valid_losses["bce"],
                valid_losses["gmm"],
                valid_losses["mse"],
            )
        )
        latest_loss = valid_loss_history[-1]["loss"]
        recent_valid_loss_hist = valid_loss_history[-1 - early_stopping_patience : -1]
        if len(valid_loss_history) > early_stopping_patience and all(
            (latest_loss >= v["loss"] for v in recent_valid_loss_hist)
        ):
            break

    trainer.mdnrnn.mdnrnn.eval()
    test_batch = test_replay_buffer.sample_memories(
        test_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
    )
    test_losses = trainer.get_loss(
        test_batch, state_dim=gym_env.state_dim, batch_first=True
    )
    test_losses = loss_to_num(test_losses)
    logger.info(
        "Test loss: {}, bce={}, gmm={}, mse={}".format(
            test_losses["loss"],
            test_losses["bce"],
            test_losses["gmm"],
            test_losses["mse"],
        )
    )
    logger.info("Valid loss history: {}".format(valid_loss_history))
    return test_losses, valid_loss_history, trainer


def create_trainer(params: Dict, env: OpenAIGymEnvironment, use_gpu: bool):
    mdnrnn_params = MDNRNNParameters(**params["mdnrnn"])
    mdnrnn_net = MemoryNetwork(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_hiddens=mdnrnn_params.hidden_size,
        num_hidden_layers=mdnrnn_params.num_hidden_layers,
        num_gaussians=mdnrnn_params.num_gaussians,
    )
    if use_gpu and torch.cuda.is_available():
        mdnrnn_net = mdnrnn_net.cuda()

    cum_loss_hist_len = (
        params["run_details"]["num_train_episodes"]
        * params["run_details"]["max_steps"]
        // mdnrnn_params.minibatch_size
    )
    trainer = MDNRNNTrainer(
        mdnrnn_network=mdnrnn_net, params=mdnrnn_params, cum_loss_hist=cum_loss_hist_len
    )
    return trainer


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
