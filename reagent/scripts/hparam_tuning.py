#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import logging  # isort:skip

logging.disable()  # isort:skip

import copy
import json
import os
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np
import torch.multiprocessing as mp
from ax.service.ax_client import AxClient


def ax_evaluate_params(
    params_list: List[Dict],
    fixed_params: Dict,
    eval_fn: Callable,
    parse_params_fn: Optional[Callable] = None,
    num_seeds: int = 10,
    num_proc: int = 20,
) -> List[Dict[str, Tuple[float, float]]]:
    """
    Evaluate a single set of hyperparameters for Ax search.

    Args:
        params_list: A list of hyperparameter configs to evaluate.
        fixed_params: A dictionary of hyperparameters that are held fixed between evaluations.
        eval_fn: Evaluation function that returns a dictionary of metric values.
        parse_params_fn: A optional function applied to the hyperparameter dictionary to parse some elements. Can be useful
            if the best representation for Ax doesn't match the format accepted by the eval_fn.
        num_seeds: Number of random seeds among which the metrics are averaged.
        num_proc: Number of processes to run in parallel.
    Returns:
        A list of average evaluation metrics (one per config)
    """
    # create a list of full hyperparameter configurations to be evaluated
    params_with_seed_list = []
    for params in params_list:
        for s in range(num_seeds):
            params_s = copy.deepcopy(params)
            params_s.update(fixed_params)
            params_s["seed"] = s
            if parse_params_fn is not None:
                params_s = parse_params_fn(params_s)
            params_with_seed_list.append(params_s)

    # evaluate metrics in parallel using multiprocessing
    if num_proc > 1:
        with mp.get_context("spawn").Pool(
            min(len(params_with_seed_list), num_proc)
        ) as p:
            metrics = p.map(eval_fn, params_with_seed_list)
    else:
        metrics = list(map(eval_fn, params_with_seed_list))

    # calculate the average metrics across different seeds
    avg_metrics = []
    num_params = len(params_list)
    for i in range(num_params):
        avg_metrics.append(
            {
                k: (
                    np.mean(
                        [m[k] for m in metrics[i * num_seeds : (i + 1) * num_seeds]]
                    ),
                    np.std(
                        [m[k] for m in metrics[i * num_seeds : (i + 1) * num_seeds]]
                    ),
                )
                for k in metrics[0].keys()
            }
        )
    return avg_metrics


def run_ax_search(
    fixed_params: Dict,
    ax_params: List[Dict[str, Any]],
    eval_fn: Callable,
    obj_name: str,
    minimize: bool,
    id_: str,
    parse_params_fn: Optional[Callable] = None,
    ax_param_constraints: Optional[List[str]] = None,
    num_ax_steps: int = 50,
    num_concur_samples: int = 2,
    num_seeds: int = 10,
    num_proc: int = 20,
    folder_name: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], AxClient]:
    """
    Run a search for best hyperparameter values using Ax.
    Note that this requires the Ax package (https://ax.dev/) to be installed.

    Args:
        fixed_params: Fixed values of hyperparameters.
        ax_params: Ax configuration for hyperparameters that are searched over. See docs for ax_client.create_experiment()
        eval_fn: Evaluation function that returns a dictionary of metric values.
        obj_name: Objective name (key of the dict returned by eval_fn)
        minimize: If True, objective is minimized, if False it's maximized.
        id_: An arbitrary string identifier of the search (used as part of filename where results are saved)
        parse_params_fn: A function applied to the parameter dictionary to parse it. Can be used
            if the best representation for Ax doesn't match the format accepted by the eval_fn.
        ax_param_constraints: Constraints for the parameters that are searched over.
        num_ax_steps: The number of ax steps to take.
        num_concur_samples: Number of configurations to sample per ax step (in parallel)
        num_seeds: Number of seeds to average over
        num_proc: Number of processes to run in parallel.
        folder_name: Folder where to save best found parameters
        verbose: If True, some details are printed out
    Returns:
        A dict of best hyperparameters found by Ax
    """
    for p in ax_params:
        assert (
            p["name"] not in fixed_params
        ), f'Parameter {p["name"]} appers in both fixed and search parameters'
    if ax_param_constraints is None:
        ax_param_constraints = []
    ax_client = AxClient()
    ax_client.create_experiment(
        name=f"hparams_search_{id_}",
        parameters=ax_params,
        objective_name=obj_name,
        minimize=minimize,
        parameter_constraints=ax_param_constraints,
        choose_generation_strategy_kwargs={
            "max_parallelism_override": num_concur_samples,
            "num_initialization_trials": max(num_concur_samples, 5, len(ax_params)),
        },
    )
    best_params = None
    all_considered_params = []
    all_considered_metrics = []

    try:
        for i in range(1, num_ax_steps + 1):
            if verbose:
                print(f"ax step {i}/{num_ax_steps}")
            params_list = []
            trial_indices_list = []
            for _ in range(num_concur_samples):
                # sample several values (to be evaluated in parallel)
                parameters, trial_index = ax_client.get_next_trial()
                params_list.append(parameters)
                trial_indices_list.append(trial_index)
            res = ax_evaluate_params(
                params_list,
                fixed_params=fixed_params,
                eval_fn=eval_fn,
                parse_params_fn=parse_params_fn,
                num_seeds=num_seeds,
                num_proc=num_proc,
            )
            all_considered_params.extend(params_list)
            all_considered_metrics.extend(res)
            for t_i, v in zip(trial_indices_list, res):
                ax_client.complete_trial(trial_index=t_i, raw_data=v)
            best_params, predicted_metrics = ax_client.get_best_parameters()
            predicted_metrics = predicted_metrics[0]  # choose expected metric values
            if verbose:
                print(best_params, predicted_metrics)
            # save at every iteration in case search is interrupted
            if folder_name is not None:
                with open(
                    os.path.join(
                        os.path.expanduser(folder_name),
                        f"ax_results_{id_}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "best_params": best_params,
                            "predicted_metrics": predicted_metrics,
                            "fixed_params": fixed_params,
                            "ax_params": ax_params,
                            "num_ax_steps": i,
                            "num_concur_samples": num_concur_samples,
                            "num_seeds": num_seeds,
                            "num_proc": num_proc,
                            "all_considered_params": all_considered_params,
                            "all_considered_metrics": all_considered_metrics,
                        },
                        f,
                        indent=4,
                    )
    except KeyboardInterrupt:
        # handle keyboard interruption to enable returning intermediate results if interrupted
        pass
    return best_params, ax_client
