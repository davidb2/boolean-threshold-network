#!/usr/bin/env python3
'''
Per-network random node selection for fixed-targets v7.

For each feature size, draws NUM_TRIALS uniformly random subsets of nodes
and evaluates drug classification accuracy (Random Forest).

Usage (one network at a time, same interface as genetic-algorithm-selection.py):
  python scripts/random-node-selection.py \
    --original-network-idx 0 \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --network-size 5000 \
    --num-workers 8 \
    --output-dir data/drug-fixed-targets-v7/N5000/random-results-v7
'''
import argparse
import math
import multiprocessing
import numpy as np
import pandas as pd
import pathlib
import random

from typing import *

from classifier import train_and_test

NUM_TRIALS = 10


def _score_one(args: Tuple) -> float:
    features, original_network_idx, particular_states_df = args
    return train_and_test(
        particular_states_df,
        num_trials=1,
        original_network_idx=original_network_idx,
        dep_vars=list(features),
    )[0]['correct'].mean()


def get_accuracies(
    particular_states_df: pd.DataFrame,
    original_network_idx: int,
    network_size: int,
    pool,
    feature_sizes: Optional[List[int]] = None,
    num_trials: int = NUM_TRIALS,
) -> List[Tuple]:
    particular_states_df = particular_states_df.drop(
        columns=['original_network_idx', 'initial_condition_idx'],
    )

    all_nodes = [f'node-{i}' for i in range(network_size)]

    if feature_sizes is None:
        powers = [2**i for i in range(int(math.log(network_size, 2)) + 1)]
        feature_sizes = powers if powers[-1] == network_size else powers + [network_size]

    rows = []
    for max_num_features in feature_sizes:
        tasks = [
            (random.sample(all_nodes, max_num_features), original_network_idx, particular_states_df)
            for _ in range(num_trials)
        ]
        accuracies = pool.map(_score_one, tasks)
        for trial, accuracy in enumerate(accuracies):
            rows.append((original_network_idx, max_num_features, trial, accuracy))
            print(f'k={max_num_features}, trial={trial}, accuracy={accuracy:.4f}', flush=True)

    return rows


def main(args: argparse.Namespace):
    output_path = pathlib.Path(f'{args.output_dir}/{args.original_network_idx}-full.csv')
    done_path = pathlib.Path(f'{args.output_dir}/{args.original_network_idx}-full.done')
    if done_path.exists():
        return

    states_df = pd.read_csv(args.states_file, index_col=0)
    states_df = states_df.reset_index().rename(columns={"drug_name": "Drug"})

    output_path.unlink(missing_ok=True)

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        rows = get_accuracies(
            original_network_idx=args.original_network_idx,
            particular_states_df=states_df[states_df['original_network_idx'] == args.original_network_idx],
            network_size=args.network_size,
            pool=pool,
            feature_sizes=args.feature_sizes,
            num_trials=args.num_trials,
        )

    pd.DataFrame(
        rows,
        columns=['original_network_idx', 'max_num_features', 'trial', 'accuracy'],
    ).to_csv(output_path, index=False)

    done_path.touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-network-idx', type=int, required=True)
    parser.add_argument('--states-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--network-size', type=int, default=5000)
    parser.add_argument('--feature-sizes', type=int, nargs='+', default=None,
                        help='feature set sizes to test (default: powers of 2 up to network-size)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='multiprocessing pool size (default: os.cpu_count())')
    parser.add_argument('--num-trials', type=int, default=NUM_TRIALS,
                        help='random subsets to draw per feature size')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
