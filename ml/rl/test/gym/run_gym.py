from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import json

from ml.rl.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import\
    RLParameters, TrainingParameters, DiscreteActionModelParameters


def run(
    env,
    trainer,
    test_run_name,
    final_score_bar=None,
    num_episodes=301,
    train_every=10,
    train_after=10,
    test_every=100,
    test_after=10,
    num_train_batches=100,
    train_batch_size=1024,
    avg_over_num_episodes=100,
    render=False,
    render_every=10
):
    avg_reward_history = []

    for i in range(num_episodes):
        env.run_episode(trainer, False, render and i % render_every == 0)

        if i % train_every == 0 and i > train_after:
            for _ in range(num_train_batches):
                trainer.stream(*env.get_replay_samples(train_batch_size))
        if i == num_episodes - 1 or (i % test_every == 0 and i > test_after):
            reward_sum = 0.0
            for test_i in range(avg_over_num_episodes):
                reward_sum += env.run_episode(
                    trainer, True, render and test_i % render_every == 0
                )
            avg_rewards = round(reward_sum / avg_over_num_episodes, 2)
            print(
                "Achieved an average reward score of {} over {} iterations"
                .format(avg_rewards, avg_over_num_episodes)
            )
            avg_reward_history.append(avg_rewards)
            if final_score_bar is not None and avg_rewards > final_score_bar:
                break

    print(
        'Averaged reward history for {}:'.format(test_run_name),
        avg_reward_history
    )
    return avg_reward_history


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in openAI GYM."
    )
    parser.add_argument(
        "-p",
        "--parameters",
        help="Path to JSON parameters file"
    )
    args = parser.parse_args(args)
    with open(args.parameters, 'r') as f:
        params = json.load(f)

    env_type = params['env']
    env = OpenAIGymEnvironment(env_type)

    rl_settings = params['rl']
    training_settings = params['training']

    if env.requires_discrete_actions:
        trainer_params = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=rl_settings['reward_discount_factor'],
                target_update_rate=rl_settings['target_update_rate'],
                reward_burnin=rl_settings['reward_burnin'],
                maxq_learning=bool(rl_settings['maxq_learning'])
            ),
            training=TrainingParameters(
                layers=training_settings['layers'],
                activations=training_settings['activations'],
                minibatch_size=training_settings['minibatch_size'],
                learning_rate=training_settings['learning_rate'],
                optimizer=training_settings['optimizer'],
                gamma=training_settings['learning_rate_decay'],
            )
        )
        trainer = DiscreteActionTrainer(
            env.normalization, trainer_params
        )
    else:
        raise Exception("Unsupported env type")

    run(
        env, trainer, "{} test run".format(env_type), **params["run_details"]
    )


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        raise Exception("Usage: python run_gym.py -p <parameters_file>")
    main(args[1:])
