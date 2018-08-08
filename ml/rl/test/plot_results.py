#!/usr/bin/env python3
"""
Script to plot results from workflow.

Required Parameters:
    --outfile [-o] The output filepath to save the plot
    --directories [-d] The directories containing results to be plotted. At
    least one directory path must be given

Optional Parameters:
    --confidence-interval [-c] The number of standard deviations in the
    confidence interval. Default is 1.96, which yields a 95% confidence interval

Usage:
    python plot_results.py -o /path/to/outfile.png -d dir1
    python plot_results.py -d dir1 dir2 dir3 -o /path/to/outfile.png -c 2.58
"""

import argparse
import os

import numpy as np
from matplotlib import pyplot as plt


def is_csv(filename):
    _, extension = os.path.splitext(filename)
    return extension == ".csv"


def get_csvs(folder):
    all_csvs = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if is_csv(filepath):
            all_csvs.append(filepath)
    return all_csvs


def read_csvs(csvs):
    _REWARD = 0
    _TIMESTEP = 1
    rewards = []
    timesteps = []
    first_pass = True
    for csv in csvs:
        data = np.loadtxt(csv, delimiter=",")
        for i, row in enumerate(data):
            reward = row[_REWARD]
            timestep = row[_TIMESTEP]
            if first_pass:
                rewards.append([reward])
                timesteps.append(timestep)
            else:
                if i >= len(timesteps):
                    break
                elif timestep != timesteps[i]:
                    rewards = rewards[:i]
                    timesteps = timesteps[:i]
                    break
                else:
                    rewards[i].append(reward)
        if first_pass:
            first_pass = False

    return rewards, timesteps


def load_data(folder, stds):
    csvs = get_csvs(folder)
    rewards, timesteps = read_csvs(csvs)
    avg_rewards = [np.mean(update) for update in rewards]
    ci_rewards = [np.std(update) * stds / np.sqrt(len(update)) for update in rewards]
    return (timesteps, avg_rewards, ci_rewards)


def make_plots(dirs, outfile, stds):
    COLORS = [
        "red",
        "blue",
        "violet",
        "goldenrod",
        "indigo",
        "green",
        "chartreuse",
        "aquamarine",
        "chocolate",
        "fuchsia",
    ]
    fig = plt.figure()
    for i, folder in enumerate(dirs):
        (x, y, err) = load_data(folder, stds)
        label = os.path.basename(folder)
        color = COLORS[i % len(COLORS)]
        plt.plot(x, y, label=label, color=color)
        plt.fill_between(x, np.subtract(y, err), np.add(y, err), alpha=0.1, color=color)
        plt.legend(loc=0)
    fig.savefig(outfile)
    plt.close()


def main(directories, outfile, stds):
    make_plots(directories, outfile, stds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--outfile", "-o", dest="outfile", type=str, help="Output file to save graph"
    )
    required.add_argument(
        "--directories",
        "-d",
        dest="directories",
        type=str,
        nargs="*",
        help="Directories containing results to be plotted",
    )
    parser.add_argument(
        "--confidence-interval",
        "-c",
        dest="confidence_interval",
        default=1.96,
        type=float,
        help="Number of standard deviations for confidence intervals",
    )
    args = parser.parse_args()

    assert args.outfile, "No outfile given. See -h for help"
    assert args.directories, "No input directories given. See -h for help"

    # Remove trailing slashes for directory paths
    directories = [dir.rstrip("/") for dir in args.directories]
    outfile = args.outfile
    stds = args.confidence_interval

    main(directories, outfile, stds)
