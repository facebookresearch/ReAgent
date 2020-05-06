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

import numpy as np
import reagent.types as rlt
import torch
from reagent.evaluation.world_model_evaluator import (
    FeatureImportanceEvaluator,
    FeatureSensitivityEvaluator,
)
from reagent.json_serialize import json_to_object
from reagent.models.mdn_rnn import MDNRNNMemoryPool
from reagent.models.world_model import MemoryNetwork
from reagent.test.gym.open_ai_gym_environment import (
    EnvType,
    ModelType,
    OpenAIGymEnvironment,
)
from reagent.test.gym.run_gym import (
    OpenAiGymParameters,
    OpenAiRunDetails,
    dict_to_np,
    get_possible_actions,
)
from reagent.training.rl_dataset import RLDataset
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)


def loss_to_num(losses):
    return {k: v.item() for k, v in losses.items()}


def multi_step_sample_generator(
    gym_env: OpenAIGymEnvironment,
    num_transitions: int,
    max_steps: Optional[int],
    multi_steps: int,
    include_shorter_samples_at_start: bool,
    include_shorter_samples_at_end: bool,
):
    """
    Convert gym env multi-step sample format to mdn-rnn multi-step sample format

    :param gym_env: The environment used to generate multi-step samples
    :param num_transitions: # of samples to return
    :param max_steps: An episode terminates when the horizon is beyond max_steps
    :param multi_steps: # of steps of states and actions per sample
    :param include_shorter_samples_at_start: Whether to keep samples of shorter steps
        which are generated at the beginning of an episode
    :param include_shorter_samples_at_end: Whether to keep samples of shorter steps
        which are generated at the end of an episode
    """
    samples = gym_env.generate_random_samples(
        num_transitions=num_transitions,
        use_continuous_action=True,
        max_step=max_steps,
        multi_steps=multi_steps,
        include_shorter_samples_at_start=include_shorter_samples_at_start,
        include_shorter_samples_at_end=include_shorter_samples_at_end,
    )

    for j in range(num_transitions):
        # pyre-fixme[6]: Expected `Sized` for 1st param but got
        #  `Union[typing.List[bool], bool]`.
        sample_steps = len(samples.terminals[j])
        state = dict_to_np(samples.states[j], np_size=gym_env.state_dim, key_offset=0)
        action = dict_to_np(
            samples.actions[j], np_size=gym_env.action_dim, key_offset=gym_env.state_dim
        )
        next_actions = np.float32(
            # pyre-fixme[6]: Expected `Union[typing.SupportsFloat,
            #  typing.SupportsInt]` for 1st param but got `List[typing.Any]`.
            [
                dict_to_np(
                    samples.next_actions[j][k],
                    np_size=gym_env.action_dim,
                    key_offset=gym_env.state_dim,
                )
                for k in range(sample_steps)
            ]
        )
        next_states = np.float32(
            # pyre-fixme[6]: Expected `Union[typing.SupportsFloat,
            #  typing.SupportsInt]` for 1st param but got `List[typing.Any]`.
            [
                dict_to_np(
                    samples.next_states[j][k], np_size=gym_env.state_dim, key_offset=0
                )
                for k in range(sample_steps)
            ]
        )
        # pyre-fixme[6]: Expected `Union[typing.SupportsFloat, typing.SupportsInt]`
        #  for 1st param but got `Union[typing.List[float], float]`.
        rewards = np.float32(samples.rewards[j])
        # pyre-fixme[6]: Expected `Union[typing.SupportsFloat, typing.SupportsInt]`
        #  for 1st param but got `Union[typing.List[bool], bool]`.
        terminals = np.float32(samples.terminals[j])
        not_terminals = np.logical_not(terminals)
        ordered_states = np.vstack((state, next_states))
        ordered_actions = np.vstack((action, next_actions))
        mdnrnn_states = ordered_states[:-1]
        mdnrnn_actions = ordered_actions[:-1]
        mdnrnn_next_states = ordered_states[-multi_steps:]
        mdnrnn_next_actions = ordered_actions[-multi_steps:]

        # Padding zeros so that all samples have equal steps
        # The general rule is to pad zeros at the end of sequences.
        # In addition, if the sequence only has one step (i.e., the
        # first state of an episode), pad one zero row ahead of the
        # sequence, which enables embedding generated properly for
        # one-step samples
        num_padded_top_rows = 1 if multi_steps > 1 and sample_steps == 1 else 0
        num_padded_bottom_rows = multi_steps - sample_steps - num_padded_top_rows
        sample_steps_next = len(mdnrnn_next_states)
        num_padded_top_rows_next = 0
        num_padded_bottom_rows_next = multi_steps - sample_steps_next
        yield (
            np.pad(
                mdnrnn_states,
                ((num_padded_top_rows, num_padded_bottom_rows), (0, 0)),
                "constant",
                constant_values=0.0,
            ),
            np.pad(
                mdnrnn_actions,
                ((num_padded_top_rows, num_padded_bottom_rows), (0, 0)),
                "constant",
                constant_values=0.0,
            ),
            np.pad(
                rewards,
                ((num_padded_top_rows, num_padded_bottom_rows)),
                "constant",
                constant_values=0.0,
            ),
            np.pad(
                mdnrnn_next_states,
                ((num_padded_top_rows_next, num_padded_bottom_rows_next), (0, 0)),
                "constant",
                constant_values=0.0,
            ),
            np.pad(
                mdnrnn_next_actions,
                ((num_padded_top_rows_next, num_padded_bottom_rows_next), (0, 0)),
                "constant",
                constant_values=0.0,
            ),
            np.pad(
                not_terminals,
                ((num_padded_top_rows, num_padded_bottom_rows)),
                "constant",
                constant_values=0.0,
            ),
            sample_steps,
            sample_steps_next,
        )


def get_replay_buffer(
    num_episodes: int, seq_len: int, max_step: int, gym_env: OpenAIGymEnvironment
) -> MDNRNNMemoryPool:
    num_transitions = num_episodes * max_step
    replay_buffer = MDNRNNMemoryPool(max_replay_memory_size=num_transitions)
    for (
        mdnrnn_state,
        mdnrnn_action,
        rewards,
        next_states,
        _,
        not_terminals,
        _,
        _,
    ) in multi_step_sample_generator(
        gym_env,
        num_transitions=num_transitions,
        max_steps=max_step,
        multi_steps=seq_len,
        include_shorter_samples_at_start=False,
        include_shorter_samples_at_end=False,
    ):
        mdnrnn_state, mdnrnn_action, next_states, rewards, not_terminals = (
            torch.tensor(mdnrnn_state),
            torch.tensor(mdnrnn_action),
            torch.tensor(next_states),
            torch.tensor(rewards),
            torch.tensor(not_terminals),
        )
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
        default=-1,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
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

    logger.setLevel(getattr(logging, args.log_level.upper()))

    with open(args.parameters, "r") as f:
        params = json_to_object(f.read(), OpenAiGymParameters)
    if args.gpu_id != -1:
        params = params._replace(use_gpu=True)

    mdnrnn_gym(
        params,
        args.feature_importance,
        args.feature_sensitivity,
        args.save_embedding_to_path,
    )


def mdnrnn_gym(
    params: OpenAiGymParameters,
    feature_importance: bool = False,
    feature_sensitivity: bool = False,
    save_embedding_to_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    assert params.mdnrnn is not None
    use_gpu = params.use_gpu
    logger.info("Running gym with params")
    logger.info(params)

    env_type = params.env
    env = OpenAIGymEnvironment(
        env_type, epsilon=1.0, softmax_policy=False, gamma=0.99, random_seed=seed
    )

    # create test data once
    assert params.run_details.max_steps is not None
    test_replay_buffer = get_replay_buffer(
        params.run_details.num_test_episodes,
        params.run_details.seq_len,
        params.run_details.max_steps,
        env,
    )
    test_batch = test_replay_buffer.sample_memories(
        test_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
    )

    trainer = create_trainer(params, env, use_gpu)
    _, _, trainer = train_sgd(
        env,
        trainer,
        use_gpu,
        "{} test run".format(env_type),
        params.mdnrnn.minibatch_size,
        params.run_details,
        test_batch=test_batch,
    )
    feature_importance_map, feature_sensitivity_map, dataset = None, None, None
    if feature_importance:
        feature_importance_map = calculate_feature_importance(
            env, trainer, use_gpu, params.run_details, test_batch=test_batch
        )
    if feature_sensitivity:
        feature_sensitivity_map = calculate_feature_sensitivity_by_actions(
            env, trainer, use_gpu, params.run_details, test_batch=test_batch
        )
    if save_embedding_to_path:
        dataset = RLDataset(save_embedding_to_path)
        create_embed_rl_dataset(env, trainer, dataset, use_gpu, params.run_details)
        dataset.save()
    return env, trainer, feature_importance_map, feature_sensitivity_map, dataset


def calculate_feature_importance(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    run_details: OpenAiRunDetails,
    test_batch: rlt.PreprocessedTrainingBatch,
):
    assert run_details.max_steps is not None
    assert run_details.num_test_episodes is not None
    assert run_details.seq_len is not None
    feature_importance_evaluator = FeatureImportanceEvaluator(
        trainer,
        discrete_action=gym_env.action_type == EnvType.DISCRETE_ACTION,
        state_feature_num=gym_env.state_dim,
        action_feature_num=gym_env.action_dim,
        sorted_action_feature_start_indices=list(range(gym_env.action_dim)),
        sorted_state_feature_start_indices=list(range(gym_env.state_dim)),
    )
    feature_loss_vector = feature_importance_evaluator.evaluate(test_batch)[
        "feature_loss_increase"
    ]
    feature_importance_map = {}
    for i in range(gym_env.action_dim):
        print(
            "action {}, feature importance: {}".format(i, feature_loss_vector[i].item())
        )
        feature_importance_map[f"action{i}"] = feature_loss_vector[i].item()
    for i in range(gym_env.state_dim):
        print(
            "state {}, feature importance: {}".format(
                i, feature_loss_vector[i + gym_env.action_dim].item()
            )
        )
        feature_importance_map[f"state{i}"] = feature_loss_vector[
            i + gym_env.action_dim
        ].item()
    return feature_importance_map


def create_embed_rl_dataset(
    gym_env: OpenAIGymEnvironment,
    trainer: MDNRNNTrainer,
    dataset: RLDataset,
    use_gpu: bool,
    run_details: OpenAiRunDetails,
):
    assert run_details.max_steps is not None
    old_mdnrnn_mode = trainer.mdnrnn.mdnrnn.training
    trainer.mdnrnn.mdnrnn.eval()
    num_transitions = run_details.num_state_embed_episodes * run_details.max_steps
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        next_action_batch,
        not_terminal_batch,
        step_batch,
        next_step_batch,
    ) = map(
        list,
        zip(
            *multi_step_sample_generator(
                gym_env=gym_env,
                num_transitions=num_transitions,
                max_steps=run_details.max_steps,
                # +1 because MDNRNN embeds the first seq_len steps and then
                # the embedded state will be concatenated with the last step
                multi_steps=run_details.seq_len + 1,
                include_shorter_samples_at_start=True,
                include_shorter_samples_at_end=False,
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

    mdnrnn_input = rlt.PreprocessedStateAction.from_tensors(
        state=mdnrnn_state, action=mdnrnn_action
    )
    next_mdnrnn_input = rlt.PreprocessedStateAction.from_tensors(
        state=next_mdnrnn_state, action=next_mdnrnn_action
    )
    # batch-compute state embedding
    mdnrnn_output = trainer.mdnrnn(mdnrnn_input)
    next_mdnrnn_output = trainer.mdnrnn(next_mdnrnn_input)

    for i in range(len(state_batch)):
        # Embed the state as the hidden layer's output
        # until the previous step + current state
        hidden_idx = 0 if step_batch[i] == 1 else step_batch[i] - 2
        next_hidden_idx = next_step_batch[i] - 2
        hidden_embed = (
            mdnrnn_output.all_steps_lstm_hidden[hidden_idx, i, :]
            .squeeze()
            .detach()
            .cpu()
        )
        state_embed = torch.cat(
            (hidden_embed, torch.tensor(state_batch[i][hidden_idx + 1]))
        )
        next_hidden_embed = (
            next_mdnrnn_output.all_steps_lstm_hidden[next_hidden_idx, i, :]
            .squeeze()
            .detach()
            .cpu()
        )
        next_state_embed = torch.cat(
            (next_hidden_embed, torch.tensor(next_state_batch[i][next_hidden_idx + 1]))
        )

        logger.debug(
            "create_embed_rl_dataset:\nstate batch\n{}\naction batch\n{}\nlast "
            "action: {},reward: {}\nstate embed {}\nnext state embed {}\n".format(
                state_batch[i][: hidden_idx + 1],
                action_batch[i][: hidden_idx + 1],
                action_batch[i][hidden_idx + 1],
                reward_batch[i][hidden_idx + 1],
                state_embed,
                next_state_embed,
            )
        )

        terminal = 1 - not_terminal_batch[i][hidden_idx + 1]
        possible_actions, possible_actions_mask = get_possible_actions(
            gym_env, ModelType.PYTORCH_PARAMETRIC_DQN.value, False
        )
        possible_next_actions, possible_next_actions_mask = get_possible_actions(
            gym_env, ModelType.PYTORCH_PARAMETRIC_DQN.value, terminal
        )
        dataset.insert(
            state=state_embed,
            action=torch.tensor(action_batch[i][hidden_idx + 1]),
            reward=float(reward_batch[i][hidden_idx + 1]),
            next_state=next_state_embed,
            next_action=torch.tensor(next_action_batch[i][next_hidden_idx + 1]),
            terminal=torch.tensor(terminal),
            possible_next_actions=possible_next_actions,
            possible_next_actions_mask=possible_next_actions_mask,
            time_diff=torch.tensor(1),
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
    run_details: OpenAiRunDetails,
    test_batch: rlt.PreprocessedTrainingBatch,
    seq_len: int = 5,
    num_test_episodes: int = 100,
):
    assert run_details.max_steps is not None
    feature_sensitivity_evaluator = FeatureSensitivityEvaluator(
        trainer,
        state_feature_num=gym_env.state_dim,
        sorted_state_feature_start_indices=list(range(gym_env.state_dim)),
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
    run_details: OpenAiRunDetails,
    test_batch: rlt.PreprocessedTrainingBatch,
):
    assert run_details.max_steps is not None
    train_replay_buffer = get_replay_buffer(
        run_details.num_train_episodes,
        run_details.seq_len,
        run_details.max_steps,
        gym_env,
    )
    valid_replay_buffer = get_replay_buffer(
        run_details.num_test_episodes,
        run_details.seq_len,
        run_details.max_steps,
        gym_env,
    )
    valid_batch = valid_replay_buffer.sample_memories(
        valid_replay_buffer.memory_size, use_gpu=use_gpu, batch_first=True
    )
    valid_loss_history = []

    num_batch_per_epoch = train_replay_buffer.memory_size // minibatch_size
    logger.info(
        "Collected data {} transitions.\n"
        "Training will take {} epochs, with each epoch having {} mini-batches"
        " and each mini-batch having {} samples".format(
            train_replay_buffer.memory_size,
            run_details.train_epochs,
            num_batch_per_epoch,
            minibatch_size,
        )
    )

    for i_epoch in range(run_details.train_epochs):
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

        trainer.mdnrnn.mdnrnn.eval()
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
        recent_valid_loss_hist = valid_loss_history[
            -1 - run_details.early_stopping_patience : -1
        ]
        # earlystopping
        if len(valid_loss_history) > run_details.early_stopping_patience and all(
            (latest_loss >= v["loss"] for v in recent_valid_loss_hist)
        ):
            break

    trainer.mdnrnn.mdnrnn.eval()
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


def create_trainer(
    params: OpenAiGymParameters, env: OpenAIGymEnvironment, use_gpu: bool
):
    assert params.mdnrnn is not None
    assert params.run_details.max_steps is not None
    mdnrnn_params = params.mdnrnn
    mdnrnn_net = MemoryNetwork(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_hiddens=mdnrnn_params.hidden_size,
        num_hidden_layers=mdnrnn_params.num_hidden_layers,
        num_gaussians=mdnrnn_params.num_gaussians,
    )
    if use_gpu:
        mdnrnn_net = mdnrnn_net.cuda()

    cum_loss_hist_len = (
        params.run_details.num_train_episodes
        * params.run_details.max_steps
        // mdnrnn_params.minibatch_size
    )
    trainer = MDNRNNTrainer(
        mdnrnn_network=mdnrnn_net, params=mdnrnn_params, cum_loss_hist=cum_loss_hist_len
    )
    return trainer


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    args = sys.argv
    main(args[1:])
