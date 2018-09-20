#!/usr/bin/env python3

import argparse
import json
import logging
import sys

import numpy as np
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from ml.rl.test.gym.gym_predictor import (
    GymDDPGPredictor,
    GymDQNPredictor,
    GymDQNPredictorPytorch,
)
from ml.rl.test.gym.open_ai_gym_environment import (
    EnvType,
    ModelType,
    OpenAIGymEnvironment,
)
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.test.utils import write_lists_to_csv
from ml.rl.thrift.core.ttypes import (
    CNNParameters,
    ContinuousActionModelParameters,
    DDPGModelParameters,
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    DiscreteActionModelParameters,
    KnnParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.training.ddpg_trainer import DDPGTrainer
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.training.rl_dataset import RLDataset


logger = logging.getLogger(__name__)

USE_CPU = -1


def get_possible_next_actions(gym_env, model_type, terminal):
    if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
        possible_next_actions = [
            0 if terminal else 1 for __ in range(gym_env.action_dim)
        ]
        possible_next_actions_lengths = gym_env.action_dim
    elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
        if terminal:
            possible_next_actions = np.array([])
            possible_next_actions_lengths = 0
        else:
            possible_next_actions = np.eye(gym_env.action_dim)
            possible_next_actions_lengths = gym_env.action_dim
    elif model_type == ModelType.CONTINUOUS_ACTION.value:
        possible_next_actions = None
        possible_next_actions_lengths = 0
    return possible_next_actions, possible_next_actions_lengths


def train_sgd(
    c2_device,
    gym_env,
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
    start_saving_from_episode=0,
):
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
        start_saving_from_episode,
    )


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
    start_saving_from_episode,
):
    """Train off of dynamic set of transitions generated on-policy."""

    total_timesteps = 0
    avg_reward_history, timestep_history = [], []

    for i in range(num_episodes):
        terminal = False
        next_state = gym_env.transform_state(gym_env.env.reset())
        next_action = gym_env.policy(predictor, next_state, False)
        reward_sum = 0
        ep_timesteps = 0

        if model_type == ModelType.CONTINUOUS_ACTION.value:
            trainer.noise.clear()

        while not terminal:
            state = next_state
            action = next_action

            # Get possible actions
            possible_actions, _ = get_possible_next_actions(
                gym_env, model_type, terminal
            )

            if render:
                gym_env.env.render()

            if gym_env.action_type == EnvType.DISCRETE_ACTION:
                action_index = np.argmax(action)
                next_state, reward, terminal, _ = gym_env.env.step(action_index)
            else:
                next_state, reward, terminal, _ = gym_env.env.step(action)
            next_state = gym_env.transform_state(next_state)

            ep_timesteps += 1
            total_timesteps += 1
            next_action = gym_env.policy(predictor, next_state, False)
            reward_sum += reward

            # Get possible next actions
            (
                possible_next_actions,
                possible_next_actions_lengths,
            ) = get_possible_next_actions(gym_env, model_type, terminal)

            replay_buffer.insert_into_memory(
                np.float32(state),
                action,
                np.float32(reward),
                np.float32(next_state),
                next_action,
                terminal,
                possible_next_actions,
                possible_next_actions_lengths,
                1,
            )

            if save_timesteps_to_dataset and i >= start_saving_from_episode:
                save_timesteps_to_dataset.insert(
                    i,
                    ep_timesteps - 1,
                    state.tolist(),
                    action.tolist(),
                    reward,
                    next_state.tolist(),
                    next_action.tolist(),
                    terminal,
                    possible_actions,
                    possible_next_actions,
                    possible_next_actions_lengths,
                    1,
                    1.0,
                )

            # Training loop
            if (
                total_timesteps % train_every_ts == 0
                and total_timesteps > train_after_ts
                and len(replay_buffer.replay_memory) >= trainer.minibatch_size
            ):
                for _ in range(num_train_batches):
                    samples = replay_buffer.sample_memories(
                        trainer.minibatch_size, model_type
                    )
                    samples.set_type(trainer.dtype)
                    trainer.train(samples)

            # Evaluation loop
            if total_timesteps % test_every_ts == 0 and total_timesteps > test_after_ts:
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
                if score_bar is not None and avg_rewards > score_bar:
                    logger.info(
                        "Avg. reward history for {}: {}".format(
                            test_run_name, avg_reward_history
                        )
                    )
                    return avg_reward_history, timestep_history, trainer, predictor

            if max_steps and ep_timesteps >= max_steps:
                break

        # Always eval on last episode if previous eval loop didn't return.
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

    logger.info(
        "Avg. reward history for {}: {}".format(test_run_name, avg_reward_history)
    )
    return avg_reward_history, timestep_history, trainer, predictor


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
        "--start_saving_from_episode",
        type=int,
        help="If file_path is set, start saving episodes from this episode num.",
        default=0,
    )
    parser.add_argument(
        "-r",
        "--results_file_path",
        help="If set, save evaluation results to file.",
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
    reward_history, timestep_history, trainer, predictor = run_gym(
        params, args.score_bar, args.gpu_id, dataset, args.start_saving_from_episode
    )
    if dataset:
        dataset.save()
    if args.results_file_path:
        write_lists_to_csv(args.results_file_path, reward_history, timestep_history)
    return reward_history


def run_gym(
    params,
    score_bar,
    gpu_id,
    save_timesteps_to_dataset=None,
    start_saving_from_episode=0,
):
    logger.info("Running gym with params")
    logger.info(params)
    rl_parameters = RLParameters(**params["rl"])

    env_type = params["env"]
    env = OpenAIGymEnvironment(
        env_type,
        rl_parameters.epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
    )
    replay_buffer = OpenAIGymMemoryPool(params["max_replay_memory_size"])
    model_type = params["model_type"]

    use_gpu = gpu_id != USE_CPU
    trainer = create_trainer(params["model_type"], params, rl_parameters, use_gpu, env)
    predictor = create_predictor(trainer, model_type, use_gpu)

    c2_device = core.DeviceOption(
        caffe2_pb2.CUDA if use_gpu else caffe2_pb2.CPU, int(gpu_id)
    )
    return train_sgd(
        c2_device,
        env,
        replay_buffer,
        model_type,
        trainer,
        predictor,
        "{} test run".format(env_type),
        score_bar,
        **params["run_details"],
        save_timesteps_to_dataset=save_timesteps_to_dataset,
        start_saving_from_episode=start_saving_from_episode,
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
        trainer = DQNTrainer(trainer_params, env.normalization, use_gpu)

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
            rl=rl_parameters,
            training=training_parameters,
            knn=KnnParameters(model_type="DQN"),
            rainbow=rainbow_parameters,
        )
        trainer = ParametricDQNTrainer(
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

        trainer = DDPGTrainer(
            trainer_params,
            env.normalization,
            env.normalization_action,
            torch.from_numpy(action_range_low).unsqueeze(dim=0),
            torch.from_numpy(action_range_high).unsqueeze(dim=0),
            use_gpu,
        )

    else:
        raise NotImplementedError("Model of type {} not supported".format(model_type))

    return trainer


def create_predictor(trainer, model_type, use_gpu):
    c2_device = core.DeviceOption(caffe2_pb2.CUDA if use_gpu else caffe2_pb2.CPU)
    if model_type == ModelType.CONTINUOUS_ACTION.value:
        predictor = GymDDPGPredictor(trainer)
    elif model_type in (
        ModelType.PYTORCH_DISCRETE_DQN.value,
        ModelType.PYTORCH_PARAMETRIC_DQN.value,
    ):
        predictor = GymDQNPredictorPytorch(trainer)
    else:
        predictor = GymDQNPredictor(trainer, c2_device)
    return predictor


if __name__ == "__main__":
    args = sys.argv
    if len(args) not in [3, 5, 7, 9, 11]:
        raise Exception(
            "Usage: python run_gym.py -p <parameters_file>"
            + " [-s <score_bar>] [-g <gpu_id>] [-l <log_level>] [-f <filename>]"
        )
    main(args[1:])
