#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import json
import logging
import pickle
import sys
from copy import deepcopy

import numpy as np
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from ml.rl.models.actor import GaussianFullyConnectedActor
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.test.base.utils import write_lists_to_csv
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymSACPredictor,
)
from ml.rl.test.gym.open_ai_gym_environment import (
    EnvType,
    ModelType,
    OpenAIGymEnvironment,
)
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.thrift.core.ttypes import (
    CNNParameters,
    ContinuousActionModelParameters,
    DDPGModelParameters,
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    DiscreteActionModelParameters,
    FeedForwardParameters,
    OptimizerParameters,
    RainbowDQNParameters,
    RLParameters,
    SACModelParameters,
    SACTrainingParameters,
    TrainingParameters,
)
from ml.rl.training.ddpg_trainer import ActorNetModel, CriticNetModel, DDPGTrainer
from ml.rl.training.rl_dataset import RLDataset
from ml.rl.training.sac_trainer import SACTrainer
from ml.rl.workflow.transitional import (
    create_dqn_trainer_from_params,
    create_parametric_dqn_trainer_from_params,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

USE_CPU = -1


def dict_to_np(d, np_size, key_offset):
    x = np.zeros(np_size, dtype=np.float32)
    for key in d:
        x[key - key_offset] = d[key]
    return x


def get_possible_actions(gym_env, model_type, terminal):
    if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
        possible_next_actions = None
        possible_next_actions_mask = [
            0 if terminal else 1 for __ in range(gym_env.action_dim)
        ]
    elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
        possible_next_actions = np.eye(gym_env.action_dim)
        possible_next_actions_mask = [
            0 if terminal else 1 for __ in range(gym_env.action_dim)
        ]
    elif model_type == ModelType.CONTINUOUS_ACTION.value:
        possible_next_actions = None
        possible_next_actions_mask = None
    elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
        possible_next_actions = None
        possible_next_actions_mask = None
    else:
        raise NotImplementedError()
    return possible_next_actions, possible_next_actions_mask


def train(
    c2_device,
    gym_env,
    offline_train,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    num_episodes=301,
    max_steps=None,
    train_every_ts=100,
    train_after_ts=10,
    test_every_ts=100,
    test_after_ts=10,
    num_train_batches=1,
    avg_over_num_episodes=100,
    render=False,
    save_timesteps_to_dataset=None,
    start_saving_from_score=None,
    solved_reward_threshold=None,
    max_episodes_to_run_after_solved=None,
    stop_training_after_solved=False,
    offline_train_epochs=3,
    path_to_pickled_transitions=None,
    bcq_imitator_hyperparams=None,
):
    if offline_train:
        return train_gym_offline_rl(
            c2_device,
            gym_env,
            replay_buffer,
            model_type,
            trainer,
            predictor,
            test_run_name,
            score_bar,
            max_steps,
            avg_over_num_episodes,
            offline_train_epochs,
            path_to_pickled_transitions,
            bcq_imitator_hyperparams,
        )
    else:
        return train_gym_online_rl(
            c2_device,
            gym_env,
            replay_buffer,
            model_type,
            trainer,
            predictor,
            test_run_name,
            score_bar,
            num_episodes,
            max_steps,
            train_every_ts,
            train_after_ts,
            test_every_ts,
            test_after_ts,
            num_train_batches,
            avg_over_num_episodes,
            render,
            save_timesteps_to_dataset,
            start_saving_from_score,
            solved_reward_threshold,
            max_episodes_to_run_after_solved,
            stop_training_after_solved,
        )


def create_random_policy_offline_dataset(gym_env, replay_buffer, max_steps, model_type):
    """Generate random transitions and and load into replay buffer."""

    samples = gym_env.generate_random_samples(
        num_transitions=replay_buffer.max_replay_memory_size,
        use_continuous_action=True,
        max_step=max_steps,
    )
    policy_id = 0
    for i in range(len(samples.mdp_ids)):
        state = dict_to_np(samples.states[i], gym_env.state_dim, 0)
        action = dict_to_np(samples.actions[i], gym_env.action_dim, gym_env.state_dim)
        reward = np.float32(samples.rewards[i])
        next_state = dict_to_np(samples.next_states[i], gym_env.state_dim, 0)
        next_action = dict_to_np(
            samples.next_actions[i], gym_env.action_dim, gym_env.state_dim
        )
        terminal = samples.terminals[i]
        (possible_actions, possible_actions_mask) = get_possible_actions(
            gym_env, model_type, False
        )
        (possible_next_actions, possible_next_actions_mask) = get_possible_actions(
            gym_env, model_type, samples.terminals[i]
        )
        replay_buffer.insert_into_memory(
            state,
            action,
            reward,
            next_state,
            next_action,
            terminal,
            possible_next_actions,
            possible_next_actions_mask,
            1,
            possible_actions,
            possible_actions_mask,
            policy_id,
        )


def create_stored_policy_offline_dataset(replay_buffer, path):
    """Read transitions from pickle file and load into replay buffer."""
    with open(path, "rb") as f:
        rows = pickle.load(f)
    unique_policies = set()
    for row in rows:
        unique_policies.add(row["policy_id"])
        replay_buffer.insert_into_memory(**row)
    logger.info(
        "Transitions generated from {} different policies".format(len(unique_policies))
    )


def train_gym_offline_rl(
    c2_device,
    gym_env,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    max_steps,
    avg_over_num_episodes,
    offline_train_epochs,
    path_to_pickled_transitions,
    bcq_imitator_hyper_params,
):
    """
    Train on transitions generated from a random policy live or
    read transitions from a pickle file and load into replay buffer.
    """
    if path_to_pickled_transitions is not None:
        logger.info("Loading transitions from {}".format(path_to_pickled_transitions))
        create_stored_policy_offline_dataset(replay_buffer, path_to_pickled_transitions)
    else:
        logger.info("Generating {} transitions under random policy.".format(max_steps))
        create_random_policy_offline_dataset(
            gym_env, replay_buffer, max_steps, model_type
        )

    num_batch_per_epoch = replay_buffer.size // trainer.minibatch_size
    logger.info(
        "{} offline transitions in replay buffer.\n"
        "Training will take {} epochs, with each epoch having {} mini-batches"
        " and each mini-batch having {} samples".format(
            replay_buffer.size,
            offline_train_epochs,
            num_batch_per_epoch,
            trainer.minibatch_size,
        )
    )

    avg_reward_history, epoch_history = [], []

    # Pre-train a GBDT imitator if doing batch constrained q-learning in Gym
    if trainer.bcq:
        gbdt = GradientBoostingClassifier(
            n_estimators=bcq_imitator_hyper_params["gbdt_trees"],
            max_depth=bcq_imitator_hyper_params["max_depth"],
        )
        samples = replay_buffer.sample_memories(replay_buffer.size, model_type)
        X, y = samples.states.numpy(), torch.max(samples.actions, dim=1)[1].numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        logger.info("Fitting GBDT...")
        gbdt.fit(X_train, y_train)
        train_score = round(gbdt.score(X_train, y_train) * 100, 1)
        test_score = round(gbdt.score(X_test, y_test) * 100, 1)
        logger.info(
            "GBDT train accuracy {}% || test accuracy {}%".format(
                train_score, test_score
            )
        )
        trainer.bcq_imitator = gbdt.predict_proba

    # Offline training
    for i_epoch in range(offline_train_epochs):
        avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
            avg_over_num_episodes, predictor, test=True
        )
        avg_reward_history.append(avg_rewards)

        # For offline training, use epoch number as timestep history since
        # we have a fixed batch of data to count epochs over.
        epoch_history.append(i_epoch)
        logger.info(
            "Achieved an average reward score of {} over {} evaluations"
            " after epoch {}.".format(avg_rewards, avg_over_num_episodes, i_epoch)
        )
        if score_bar is not None and avg_rewards > score_bar:
            logger.info(
                "Avg. reward history for {}: {}".format(
                    test_run_name, avg_reward_history
                )
            )
            return avg_reward_history, epoch_history, trainer, predictor

        for _ in range(num_batch_per_epoch):
            samples = replay_buffer.sample_memories(trainer.minibatch_size, model_type)
            samples.set_type(trainer.dtype)
            trainer.train(samples)

        batch_td_loss = np.mean(
            [stat.td_loss for stat in trainer.loss_reporter.incoming_stats]
        )
        trainer.loss_reporter.flush()
        logger.info(
            "Average TD loss: {} in epoch {}".format(batch_td_loss, i_epoch + 1)
        )

    logger.info(
        "Avg. reward history for {}: {}".format(test_run_name, avg_reward_history)
    )
    return avg_reward_history, epoch_history, trainer, predictor, gym_env


def train_gym_online_rl(
    c2_device,
    gym_env,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    num_episodes,
    max_steps,
    train_every_ts,
    train_after_ts,
    test_every_ts,
    test_after_ts,
    num_train_batches,
    avg_over_num_episodes,
    render,
    save_timesteps_to_dataset,
    start_saving_from_score,
    solved_reward_threshold,
    max_episodes_to_run_after_solved,
    stop_training_after_solved,
):
    """Train off of dynamic set of transitions generated on-policy."""
    total_timesteps = 0
    avg_reward_history, timestep_history = [], []
    best_episode_score_seeen = -1e20
    episodes_since_solved = 0
    solved = False
    policy_id = 0

    for i in range(num_episodes):
        if (
            max_episodes_to_run_after_solved is not None
            and episodes_since_solved > max_episodes_to_run_after_solved
        ):
            break

        if solved:
            episodes_since_solved += 1

        terminal = False
        next_state = gym_env.transform_state(gym_env.env.reset())
        next_action, next_action_probability = gym_env.policy(
            predictor, next_state, False
        )
        reward_sum = 0
        ep_timesteps = 0

        if model_type == ModelType.CONTINUOUS_ACTION.value:
            trainer.noise.clear()

        while not terminal:
            state = next_state
            action = next_action
            action_probability = next_action_probability

            # Get possible actions
            possible_actions, _ = get_possible_actions(gym_env, model_type, terminal)

            if render:
                gym_env.env.render()

            timeline_format_action, gym_action = _format_action_for_log_and_gym(
                action, gym_env.action_type, model_type
            )
            next_state, reward, terminal, _ = gym_env.env.step(gym_action)
            next_state = gym_env.transform_state(next_state)

            ep_timesteps += 1
            total_timesteps += 1
            next_action, next_action_probability = gym_env.policy(
                predictor, next_state, False
            )
            reward_sum += reward

            (possible_actions, possible_actions_mask) = get_possible_actions(
                gym_env, model_type, False
            )

            # Get possible next actions
            (possible_next_actions, possible_next_actions_mask) = get_possible_actions(
                gym_env, model_type, terminal
            )

            replay_buffer.insert_into_memory(
                np.float32(state),
                action,
                np.float32(reward),
                np.float32(next_state),
                next_action,
                terminal,
                possible_next_actions,
                possible_next_actions_mask,
                1,
                possible_actions,
                possible_actions_mask,
                policy_id,
            )

            if save_timesteps_to_dataset and (
                start_saving_from_score is None
                or best_episode_score_seeen >= start_saving_from_score
            ):
                save_timesteps_to_dataset.insert(
                    mdp_id=i,
                    sequence_number=ep_timesteps - 1,
                    state=state,
                    action=action,
                    timeline_format_action=timeline_format_action,
                    action_probability=action_probability,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    terminal=terminal,
                    possible_next_actions=possible_next_actions,
                    possible_next_actions_mask=possible_next_actions_mask,
                    time_diff=1,
                    possible_actions=possible_actions,
                    possible_actions_mask=possible_actions_mask,
                    policy_id=policy_id,
                )

            # Training loop
            if (
                total_timesteps % train_every_ts == 0
                and total_timesteps > train_after_ts
                and len(replay_buffer.replay_memory) >= trainer.minibatch_size
                and not (stop_training_after_solved and solved)
            ):
                for _ in range(num_train_batches):
                    samples = replay_buffer.sample_memories(
                        trainer.minibatch_size, model_type
                    )
                    samples.set_type(trainer.dtype)
                    trainer.train(samples)
                    # Every time we train, the policy changes
                    policy_id += 1

            # Evaluation loop
            if total_timesteps % test_every_ts == 0 and total_timesteps > test_after_ts:
                avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
                    avg_over_num_episodes, predictor, test=True
                )
                if avg_rewards > best_episode_score_seeen:
                    best_episode_score_seeen = avg_rewards

                if (
                    solved_reward_threshold is not None
                    and best_episode_score_seeen > solved_reward_threshold
                ):
                    solved = True

                avg_reward_history.append(avg_rewards)
                timestep_history.append(total_timesteps)
                logger.info(
                    "Achieved an average reward score of {} over {} evaluations."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
                    )
                )
                if score_bar is not None and avg_rewards > score_bar:
                    logger.info(
                        "Avg. reward history for {}: {}".format(
                            test_run_name, avg_reward_history
                        )
                    )
                    return (
                        avg_reward_history,
                        timestep_history,
                        trainer,
                        predictor,
                        gym_env,
                    )

            if max_steps and ep_timesteps >= max_steps:
                break

        # Always eval on last episode
        if i == num_episodes - 1:
            avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
                avg_over_num_episodes, predictor, test=True
            )
            avg_reward_history.append(avg_rewards)
            timestep_history.append(total_timesteps)
            logger.info(
                "Achieved an average reward score of {} over {} evaluations."
                " Total episodes: {}, total timesteps: {}.".format(
                    avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
                )
            )

        if solved:
            gym_env.epsilon = gym_env.minimum_epsilon
        else:
            gym_env.decay_epsilon()

    logger.info(
        "Avg. reward history for {}: {}".format(test_run_name, avg_reward_history)
    )
    return avg_reward_history, timestep_history, trainer, predictor, gym_env


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment."
    )
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    parser.add_argument(
        "-s",
        "--score-bar",
        help="Bar for averaged tests scores.",
        type=float,
        default=None,
    )
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
        "--file_path",
        help="If set, save all collected samples as an RLDataset to this file.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--start_saving_from_score",
        type=int,
        help="If file_path is set, start saving episodes after this score is hit.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--results_file_path",
        help="If set, save evaluation results to file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--offline_train",
        action="store_true",
        help="If set, collect data using a random policy then train RL offline.",
    )
    parser.add_argument(
        "--path_to_pickled_transitions",
        help="Path to saved transitions to load into replay buffer.",
        type=str,
        default=None,
    )
    args = parser.parse_args(args)

    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception("Logging level {} not valid level.".format(args.log_level))
    else:
        logger.setLevel(getattr(logging, args.log_level.upper()))

    with open(args.parameters, "r") as f:
        params = json.load(f)

    dataset = RLDataset(args.file_path) if args.file_path else None

    reward_history, iteration_history, trainer, predictor, env = run_gym(
        params,
        args.offline_train,
        args.score_bar,
        args.gpu_id,
        dataset,
        args.start_saving_from_score,
        args.path_to_pickled_transitions,
    )

    if dataset:
        dataset.save()
        logger.info("Saving dataset to {}".format(args.file_path))
        final_score_exploit, _ = env.run_ep_n_times(
            params["run_details"]["avg_over_num_episodes"], predictor, test=True
        )
        final_score_explore, _ = env.run_ep_n_times(
            params["run_details"]["avg_over_num_episodes"], predictor, test=False
        )
        logger.info(
            "Final policy scores {} with ε={} and {} with ε=0 over {} eps.".format(
                final_score_explore,
                env.epsilon,
                final_score_exploit,
                params["run_details"]["avg_over_num_episodes"],
            )
        )

    if args.results_file_path:
        write_lists_to_csv(args.results_file_path, reward_history, iteration_history)
    return reward_history


def run_gym(
    params,
    offline_train,
    score_bar,
    gpu_id,
    save_timesteps_to_dataset=None,
    start_saving_from_score=None,
    path_to_pickled_transitions=None,
):
    logger.info("Running gym with params")
    logger.info(params)
    rl_parameters = RLParameters(**params["rl"])

    env_type = params["env"]
    if offline_train:
        # take random actions during data collection
        epsilon = 1.0
    else:
        epsilon = rl_parameters.epsilon

    epsilon_decay, minimum_epsilon = 1.0, None
    if "epsilon_decay" in params["run_details"]:
        epsilon_decay = params["run_details"]["epsilon_decay"]
        del params["run_details"]["epsilon_decay"]
    if "minimum_epsilon" in params["run_details"]:
        minimum_epsilon = params["run_details"]["minimum_epsilon"]
        del params["run_details"]["minimum_epsilon"]

    env = OpenAIGymEnvironment(
        env_type,
        epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
        epsilon_decay,
        minimum_epsilon,
    )
    replay_buffer = OpenAIGymMemoryPool(params["max_replay_memory_size"])
    model_type = params["model_type"]

    use_gpu = gpu_id != USE_CPU
    trainer = create_trainer(params["model_type"], params, rl_parameters, use_gpu, env)
    predictor = create_predictor(trainer, model_type, use_gpu, env.action_dim)

    c2_device = core.DeviceOption(
        caffe2_pb2.CUDA if use_gpu else caffe2_pb2.CPU, int(gpu_id)
    )
    return train(
        c2_device,
        env,
        offline_train,
        replay_buffer,
        model_type,
        trainer,
        predictor,
        "{} test run".format(env_type),
        score_bar,
        **params["run_details"],
        save_timesteps_to_dataset=save_timesteps_to_dataset,
        start_saving_from_score=start_saving_from_score,
        path_to_pickled_transitions=path_to_pickled_transitions,
    )


def create_trainer(model_type, params, rl_parameters, use_gpu, env):
    if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
        training_parameters = params["training"]
        if isinstance(training_parameters, dict):
            training_parameters = TrainingParameters(**training_parameters)
        rainbow_parameters = params["rainbow"]
        if isinstance(rainbow_parameters, dict):
            rainbow_parameters = RainbowDQNParameters(**rainbow_parameters)
        if env.img:
            assert (
                training_parameters.cnn_parameters is not None
            ), "Missing CNN parameters for image input"
            if isinstance(training_parameters.cnn_parameters, dict):
                training_parameters.cnn_parameters = CNNParameters(
                    **training_parameters.cnn_parameters
                )
            training_parameters.cnn_parameters.conv_dims[0] = env.num_input_channels
            training_parameters.cnn_parameters.input_height = env.height
            training_parameters.cnn_parameters.input_width = env.width
            training_parameters.cnn_parameters.num_input_channels = (
                env.num_input_channels
            )
        else:
            assert (
                training_parameters.cnn_parameters is None
            ), "Extra CNN parameters for non-image input"
        trainer_params = DiscreteActionModelParameters(
            actions=env.actions,
            rl=rl_parameters,
            training=training_parameters,
            rainbow=rainbow_parameters,
        )
        trainer = create_dqn_trainer_from_params(
            trainer_params, env.normalization, use_gpu
        )

    elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
        training_parameters = params["training"]
        if isinstance(training_parameters, dict):
            training_parameters = TrainingParameters(**training_parameters)
        rainbow_parameters = params["rainbow"]
        if isinstance(rainbow_parameters, dict):
            rainbow_parameters = RainbowDQNParameters(**rainbow_parameters)
        if env.img:
            assert (
                training_parameters.cnn_parameters is not None
            ), "Missing CNN parameters for image input"
            training_parameters.cnn_parameters.conv_dims[0] = env.num_input_channels
        else:
            assert (
                training_parameters.cnn_parameters is None
            ), "Extra CNN parameters for non-image input"
        trainer_params = ContinuousActionModelParameters(
            rl=rl_parameters, training=training_parameters, rainbow=rainbow_parameters
        )
        trainer = create_parametric_dqn_trainer_from_params(
            trainer_params, env.normalization, env.normalization_action, use_gpu
        )
    elif model_type == ModelType.CONTINUOUS_ACTION.value:
        training_parameters = params["shared_training"]
        if isinstance(training_parameters, dict):
            training_parameters = DDPGTrainingParameters(**training_parameters)

        actor_parameters = params["actor_training"]
        if isinstance(actor_parameters, dict):
            actor_parameters = DDPGNetworkParameters(**actor_parameters)

        critic_parameters = params["critic_training"]
        if isinstance(critic_parameters, dict):
            critic_parameters = DDPGNetworkParameters(**critic_parameters)

        trainer_params = DDPGModelParameters(
            rl=rl_parameters,
            shared_training=training_parameters,
            actor_training=actor_parameters,
            critic_training=critic_parameters,
        )

        action_range_low = env.action_space.low.astype(np.float32)
        action_range_high = env.action_space.high.astype(np.float32)

        state_dim = get_num_output_features(env.normalization)
        action_dim = get_num_output_features(env.normalization_action)

        # Build Actor Network
        actor_network = ActorNetModel(
            layers=(
                [state_dim] + trainer_params.actor_training.layers[1:-1] + [action_dim]
            ),
            activations=trainer_params.actor_training.activations,
            fl_init=trainer_params.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=False,
        )

        # Build Critic Network
        critic_network = CriticNetModel(
            # Ensure dims match input state and scalar output
            layers=[state_dim] + trainer_params.critic_training.layers[1:-1] + [1],
            activations=trainer_params.critic_training.activations,
            fl_init=trainer_params.shared_training.final_layer_init,
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            use_all_avail_gpus=False,
        )

        trainer = DDPGTrainer(
            actor_network,
            critic_network,
            trainer_params,
            env.normalization,
            env.normalization_action,
            torch.from_numpy(action_range_low).unsqueeze(dim=0),
            torch.from_numpy(action_range_high).unsqueeze(dim=0),
            use_gpu,
        )

    elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
        trainer_params = SACModelParameters(
            rl=rl_parameters,
            training=SACTrainingParameters(
                minibatch_size=params["sac_training"]["minibatch_size"],
                use_2_q_functions=params["sac_training"]["use_2_q_functions"],
                q_network_optimizer=OptimizerParameters(
                    **params["sac_training"]["q_network_optimizer"]
                ),
                value_network_optimizer=OptimizerParameters(
                    **params["sac_training"]["value_network_optimizer"]
                ),
                actor_network_optimizer=OptimizerParameters(
                    **params["sac_training"]["actor_network_optimizer"]
                ),
                entropy_temperature=params["sac_training"]["entropy_temperature"],
            ),
            q_network=FeedForwardParameters(**params["sac_q_training"]),
            value_network=FeedForwardParameters(**params["sac_value_training"]),
            actor_network=FeedForwardParameters(**params["sac_actor_training"]),
        )
        trainer = get_sac_trainer(env, trainer_params, use_gpu)

    else:
        raise NotImplementedError("Model of type {} not supported".format(model_type))

    return trainer


def get_sac_trainer(env, parameters, use_gpu):
    trainer_args, trainer_kwargs = _get_sac_trainer_params(env, parameters, use_gpu)
    return SACTrainer(*trainer_args, **trainer_kwargs)


def _get_sac_trainer_params(env, sac_model_params, use_gpu):
    state_dim = get_num_output_features(env.normalization)
    action_dim = get_num_output_features(env.normalization_action)
    q1_network = FullyConnectedParametricDQN(
        state_dim,
        action_dim,
        sac_model_params.q_network.layers,
        sac_model_params.q_network.activations,
    )
    q2_network = None
    if sac_model_params.training.use_2_q_functions:
        q2_network = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            sac_model_params.q_network.layers,
            sac_model_params.q_network.activations,
        )
    value_network = FullyConnectedNetwork(
        [state_dim] + sac_model_params.value_network.layers + [1],
        sac_model_params.value_network.activations + ["linear"],
    )
    actor_network = GaussianFullyConnectedActor(
        state_dim,
        action_dim,
        sac_model_params.actor_network.layers,
        sac_model_params.actor_network.activations,
    )
    if use_gpu:
        q1_network.cuda()
        if q2_network:
            q2_network.cuda()
        value_network.cuda()
        actor_network.cuda()
    value_network_target = deepcopy(value_network)
    min_action_range_tensor_training = torch.full((1, action_dim), -1 + 1e-6)
    max_action_range_tensor_training = torch.full((1, action_dim), 1 - 1e-6)
    action_range_low = env.action_space.low.astype(np.float32)
    action_range_high = env.action_space.high.astype(np.float32)
    min_action_range_tensor_serving = torch.from_numpy(action_range_low).unsqueeze(
        dim=0
    )
    max_action_range_tensor_serving = torch.from_numpy(action_range_high).unsqueeze(
        dim=0
    )

    trainer_args = [
        q1_network,
        value_network,
        value_network_target,
        actor_network,
        sac_model_params,
    ]
    trainer_kwargs = {
        "q2_network": q2_network,
        "min_action_range_tensor_training": min_action_range_tensor_training,
        "max_action_range_tensor_training": max_action_range_tensor_training,
        "min_action_range_tensor_serving": min_action_range_tensor_serving,
        "max_action_range_tensor_serving": max_action_range_tensor_serving,
    }
    return trainer_args, trainer_kwargs


def _format_action_for_log_and_gym(action, env_type, model_type):
    if env_type == EnvType.DISCRETE_ACTION:
        action_index = np.argmax(action)
        if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
            return str(action_index), int(action_index)
        else:
            return action.tolist(), int(action_index)
    return action.tolist(), action.tolist()


def create_predictor(trainer, model_type, use_gpu, action_dim=None):
    if model_type == ModelType.CONTINUOUS_ACTION.value:
        predictor = GymDDPGPredictor(trainer, action_dim)
    elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
        predictor = GymSACPredictor(trainer, action_dim)
    elif model_type in (
        ModelType.PYTORCH_DISCRETE_DQN.value,
        ModelType.PYTORCH_PARAMETRIC_DQN.value,
    ):
        predictor = GymDQNPredictor(trainer, action_dim)
    else:
        raise NotImplementedError()
    return predictor


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
