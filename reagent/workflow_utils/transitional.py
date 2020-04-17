#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict

import torch
from reagent.models.actor import FullyConnectedActor
from reagent.models.categorical_dqn import CategoricalDQN
from reagent.models.cem_planner import CEMPlannerNetwork
from reagent.models.dqn import FullyConnectedDQN
from reagent.models.dueling_q_network import DuelingQNetwork
from reagent.models.parametric_dqn import FullyConnectedParametricDQN
from reagent.models.quantile_dqn import QuantileDQN
from reagent.models.world_model import MemoryNetwork
from reagent.parameters import (
    CEMParameters,
    ContinuousActionModelParameters,
    DiscreteActionModelParameters,
    MDNRNNParameters,
    OptimizerParameters,
)
from reagent.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from reagent.test.gym.open_ai_gym_environment import EnvType, OpenAIGymEnvironment
from reagent.training.c51_trainer import C51Trainer, C51TrainerParameters
from reagent.training.cem_trainer import CEMTrainer
from reagent.training.dqn_trainer import DQNTrainer, DQNTrainerParameters
from reagent.training.parametric_dqn_trainer import (
    ParametricDQNTrainer,
    ParametricDQNTrainerParameters,
)
from reagent.training.qrdqn_trainer import QRDQNTrainer, QRDQNTrainerParameters
from reagent.training.td3_trainer import TD3Trainer
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer


def create_dqn_trainer_from_params(
    model: DiscreteActionModelParameters,
    normalization_parameters: Dict[int, NormalizationParameters],
    use_gpu: bool = False,
    use_all_avail_gpus: bool = False,
    metrics_to_score=None,
):
    metrics_to_score = metrics_to_score or []

    if model.rainbow.quantile:
        q_network = QuantileDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=len(model.actions),
            num_atoms=model.rainbow.num_atoms,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )
    elif model.rainbow.categorical:
        q_network = CategoricalDQN(  # type: ignore
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=len(model.actions),
            num_atoms=model.rainbow.num_atoms,
            qmin=model.rainbow.qmin,
            qmax=model.rainbow.qmax,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
            use_gpu=use_gpu,
        )
    elif model.rainbow.dueling_architecture:
        q_network = DuelingQNetwork(  # type: ignore
            layers=[get_num_output_features(normalization_parameters)]
            + model.training.layers[1:-1]
            + [len(model.actions)],
            activations=model.training.activations,
        )
    else:
        q_network = FullyConnectedDQN(  # type: ignore
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=len(model.actions),
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )

    if use_gpu and torch.cuda.is_available():
        q_network = q_network.cuda()

    q_network_target = q_network.get_target_network()

    reward_network, q_network_cpe, q_network_cpe_target = None, None, None
    if model.evaluation.calc_cpe_in_training:
        # Metrics + reward
        num_output_nodes = (len(metrics_to_score) + 1) * len(model.actions)
        reward_network = FullyConnectedDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=num_output_nodes,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )
        q_network_cpe = FullyConnectedDQN(
            state_dim=get_num_output_features(normalization_parameters),
            action_dim=num_output_nodes,
            sizes=model.training.layers[1:-1],
            activations=model.training.activations[:-1],
            dropout_ratio=model.training.dropout_ratio,
        )

        if use_gpu and torch.cuda.is_available():
            reward_network.cuda()
            q_network_cpe.cuda()

        q_network_cpe_target = q_network_cpe.get_target_network()

    if (
        use_all_avail_gpus
        and not model.rainbow.categorical
        and not model.rainbow.quantile
    ):
        q_network = q_network.get_distributed_data_parallel_model()
        reward_network = (
            reward_network.get_distributed_data_parallel_model()
            if reward_network
            else None
        )
        q_network_cpe = (
            q_network_cpe.get_distributed_data_parallel_model()
            if q_network_cpe
            else None
        )

    if model.rainbow.quantile:
        assert (
            not use_all_avail_gpus
        ), "use_all_avail_gpus not implemented for distributional RL"
        parameters = QRDQNTrainerParameters.from_discrete_action_model_parameters(model)
        return QRDQNTrainer(
            q_network,
            q_network_target,
            parameters,
            use_gpu,
            metrics_to_score=metrics_to_score,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
        )

    elif model.rainbow.categorical:
        assert (
            not use_all_avail_gpus
        ), "use_all_avail_gpus not implemented for distributional RL"
        return C51Trainer(
            q_network,
            q_network_target,
            C51TrainerParameters.from_discrete_action_model_parameters(model),
            use_gpu,
            metrics_to_score=metrics_to_score,
        )

    else:
        parameters = DQNTrainerParameters.from_discrete_action_model_parameters(model)
        return DQNTrainer(
            q_network,
            q_network_target,
            reward_network,
            parameters,
            use_gpu,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=metrics_to_score,
        )


def create_parametric_dqn_trainer_from_params(
    model: ContinuousActionModelParameters,
    state_normalization_parameters: Dict[int, NormalizationParameters],
    action_normalization_parameters: Dict[int, NormalizationParameters],
    use_gpu: bool = False,
    use_all_avail_gpus: bool = False,
):
    q_network = FullyConnectedParametricDQN(
        state_dim=get_num_output_features(state_normalization_parameters),
        action_dim=get_num_output_features(action_normalization_parameters),
        sizes=model.training.layers[1:-1],
        activations=model.training.activations[:-1],
    )
    reward_network = FullyConnectedParametricDQN(
        state_dim=get_num_output_features(state_normalization_parameters),
        action_dim=get_num_output_features(action_normalization_parameters),
        sizes=model.training.layers[1:-1],
        activations=model.training.activations[:-1],
    )
    q_network_target = q_network.get_target_network()

    if use_gpu:
        q_network = q_network.cuda()
        q_network_target = q_network_target.cuda()
        reward_network = reward_network.cuda()

    if use_all_avail_gpus:
        q_network = q_network.get_distributed_data_parallel_model()
        q_network_target = q_network_target.get_distributed_data_parallel_model()
        reward_network = reward_network.get_distributed_data_parallel_model()

    trainer_parameters = ParametricDQNTrainerParameters(  # type: ignore
        rl=model.rl,
        double_q_learning=model.rainbow.double_q_learning,
        minibatch_size=model.training.minibatch_size,
        optimizer=OptimizerParameters(
            optimizer=model.training.optimizer,
            learning_rate=model.training.learning_rate,
            l2_decay=model.training.l2_decay,
        ),
    )

    return ParametricDQNTrainer(
        q_network,
        q_network_target,
        reward_network,
        use_gpu=use_gpu,
        **trainer_parameters.asdict()  # type: ignore
    )


def get_td3_trainer(env, parameters, use_gpu):
    state_dim = get_num_output_features(env.normalization)
    action_dim = get_num_output_features(env.normalization_action)
    q1_network = FullyConnectedParametricDQN(
        state_dim,
        action_dim,
        parameters.q_network.layers,
        parameters.q_network.activations,
    )
    q2_network = None
    if parameters.training.use_2_q_functions:
        q2_network = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            parameters.q_network.layers,
            parameters.q_network.activations,
        )
    actor_network = FullyConnectedActor(
        state_dim,
        action_dim,
        parameters.actor_network.layers,
        parameters.actor_network.activations,
    )
    min_action_range_tensor_training = torch.full((1, action_dim), -1)
    max_action_range_tensor_training = torch.full((1, action_dim), 1)
    min_action_range_tensor_serving = torch.FloatTensor(env.action_space.low).unsqueeze(
        dim=0
    )
    max_action_range_tensor_serving = torch.FloatTensor(
        env.action_space.high
    ).unsqueeze(dim=0)
    if use_gpu:
        q1_network.cuda()
        if q2_network:
            q2_network.cuda()
        actor_network.cuda()
        min_action_range_tensor_training = min_action_range_tensor_training.cuda()
        max_action_range_tensor_training = max_action_range_tensor_training.cuda()
        min_action_range_tensor_serving = min_action_range_tensor_serving.cuda()
        max_action_range_tensor_serving = max_action_range_tensor_serving.cuda()
    trainer_args = [q1_network, actor_network, parameters]
    trainer_kwargs = {
        "q2_network": q2_network,
        "min_action_range_tensor_training": min_action_range_tensor_training,
        "max_action_range_tensor_training": max_action_range_tensor_training,
        "min_action_range_tensor_serving": min_action_range_tensor_serving,
        "max_action_range_tensor_serving": max_action_range_tensor_serving,
    }
    return TD3Trainer(*trainer_args, use_gpu=use_gpu, **trainer_kwargs)


def get_cem_trainer(
    env: OpenAIGymEnvironment, params: CEMParameters, use_gpu: bool
) -> CEMTrainer:
    num_world_models = params.num_world_models
    world_model_trainers = [
        create_world_model_trainer(env, params.mdnrnn, use_gpu)
        for _ in range(num_world_models)
    ]
    world_model_nets = [trainer.mdnrnn for trainer in world_model_trainers]
    discrete_action = env.action_type == EnvType.DISCRETE_ACTION
    terminal_effective = params.mdnrnn.not_terminal_loss_weight > 0
    action_upper_bounds, action_lower_bounds = None, None
    if not discrete_action:
        action_upper_bounds, action_lower_bounds = (
            env.action_space.high,  # type: ignore
            env.action_space.low,  # type: ignore
        )

    cem_planner_network = CEMPlannerNetwork(
        mem_net_list=world_model_nets,
        cem_num_iterations=params.cem_num_iterations,
        cem_population_size=params.cem_population_size,
        ensemble_population_size=params.ensemble_population_size,
        num_elites=params.num_elites,
        plan_horizon_length=params.plan_horizon_length,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        discrete_action=discrete_action,
        terminal_effective=terminal_effective,
        gamma=params.rl.gamma,
        alpha=params.alpha,
        epsilon=params.epsilon,
        action_upper_bounds=action_upper_bounds,
        action_lower_bounds=action_lower_bounds,
    )
    cem_trainer = CEMTrainer(cem_planner_network, world_model_trainers, params, use_gpu)
    return cem_trainer


def create_world_model_trainer(
    env: OpenAIGymEnvironment, mdnrnn_params: MDNRNNParameters, use_gpu: bool
) -> MDNRNNTrainer:
    mdnrnn_net = MemoryNetwork(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_hiddens=mdnrnn_params.hidden_size,
        num_hidden_layers=mdnrnn_params.num_hidden_layers,
        num_gaussians=mdnrnn_params.num_gaussians,
    )
    if use_gpu:
        mdnrnn_net = mdnrnn_net.cuda()
    mdnrnn_trainer = MDNRNNTrainer(mdnrnn_network=mdnrnn_net, params=mdnrnn_params)
    return mdnrnn_trainer
