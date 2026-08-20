#!/usr/bin/env python3
'''Rescore the genetic algorithm's final panels with fresh evaluations.

The accuracy the search reports for itself is the best score it met while
optimizing, and the maximum of a noisy score is inflated by luck. This
script takes the final generation panel at each size and scores it again
with fresh random train and test splits of the standard evaluator, so the
number reported for the search is measured the same way as the numbers
reported for every other selection strategy.

Usage:
  python scripts/rescore-final-panels.py \
    --ga-csv data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
    --states-files data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --sizes 1 2 4 8 16 32 64 128 \
    --num-trials 10 \
    --workers 16 \
    --output-csv data/rescored/ga-rescored-rho0.5.csv
'''
import argparse
import ast
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from classifier import train_and_test

STATES = None          # loaded once in the parent, shared with workers by fork


def _load(states_files):
  global STATES
  frames = []
  for f in states_files:
    df = pd.read_csv(f, index_col=0).reset_index()
    df = df.rename(columns={'drug_name': 'Drug'})
    frames.append(df)
  STATES = pd.concat(frames, ignore_index=True)
  print(f'loaded {len(STATES)} state rows from {len(states_files)} files',
        flush=True)


def _score_network(task):
  net, panels = task
  sub = STATES[STATES.original_network_idx == net]
  drop = [c for c in ('original_network_idx', 'initial_condition_idx')
          if c in sub.columns]
  sub = sub.drop(columns=drop)
  rows = []
  for m, features in panels:
    perf, _ = train_and_test(sub, num_trials=NUM_TRIALS,
                             original_network_idx=net, dep_vars=features)
    acc = float(perf['correct'].mean())
    rows.append(dict(original_network_idx=net, max_num_features=m,
                     accuracy=acc, n_features=len(features)))
    print(f'net {net} m {m}: {acc:.3f}', flush=True)
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ga-csv', required=True)
  p.add_argument('--states-files', nargs='+', required=True)
  p.add_argument('--sizes', type=int, nargs='+',
                 default=[1, 2, 4, 8, 16, 32, 64, 128])
  p.add_argument('--num-trials', type=int, default=10)
  p.add_argument('--workers', type=int, default=8)
  p.add_argument('--output-csv', required=True)
  args = p.parse_args()

  global NUM_TRIALS
  NUM_TRIALS = args.num_trials

  ga = pd.read_csv(args.ga_csv)
  ga = ga[ga.max_num_features.isin(args.sizes)]
  fin = ga.loc[ga.groupby(['original_network_idx',
                           'max_num_features'])['generation'].idxmax()]
  tasks = []
  for net, g in fin.groupby('original_network_idx'):
    panels = [(int(r.max_num_features), sorted(ast.literal_eval(r.features)))
              for r in g.itertuples()]
    tasks.append((int(net), panels))
  print(f'{len(tasks)} networks, {sum(len(p) for _, p in tasks)} panels')

  _load(args.states_files)
  with multiprocessing.Pool(args.workers) as pool:
    all_rows = [r for rows in pool.imap_unordered(_score_network, tasks)
                for r in rows]

  out = pathlib.Path(args.output_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(all_rows).sort_values(
    ['original_network_idx', 'max_num_features']).to_csv(out, index=False)
  print(f'wrote {out} ({len(all_rows)} rows)')


NUM_TRIALS = 10

if __name__ == '__main__':
  main()
