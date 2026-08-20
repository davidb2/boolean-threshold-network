#!/usr/bin/env python3
'''Rescore the best panel of each generation on fresh trials.

If the search were budget limited, the honest accuracy of its best panel
would still be climbing when the run ends. If it is evaluation noise
limited, the honest accuracy goes flat once fitness reaches the noise
ceiling, and the continuing rise of the reported fitness is the maximum
of a noisy score harvesting luck.

Usage:
  python scripts/rescore-ga-trajectory.py \
    --ga-csv ... --states-files ... --m 8 \
    --generations 0 2 5 10 15 20 25 29 \
    --num-trials 10 --workers 16 --output-csv ...
'''
import argparse
import ast
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from classifier import train_and_test

STATES = None
NUM_TRIALS = 10


def _load(states_files):
  global STATES
  frames = []
  for f in states_files:
    df = pd.read_csv(f, index_col=0).reset_index()
    df = df.rename(columns={'drug_name': 'Drug'})
    frames.append(df)
  STATES = pd.concat(frames, ignore_index=True)
  print(f'loaded {len(STATES)} state rows', flush=True)


def _score_network(task):
  net, gens = task
  sub = STATES[STATES.original_network_idx == net]
  drop = [c for c in ('original_network_idx', 'initial_condition_idx')
          if c in sub.columns]
  sub = sub.drop(columns=drop)
  rows = []
  for gen, fitness, features in gens:
    perf, _ = train_and_test(sub, num_trials=NUM_TRIALS,
                             original_network_idx=net, dep_vars=features)
    acc = float(perf['correct'].mean())
    rows.append(dict(original_network_idx=net, generation=gen,
                     fitness=fitness, accuracy=acc))
    print(f'net {net} gen {gen}: fitness {fitness:.3f} fresh {acc:.3f}',
          flush=True)
  return rows


def main():
  global NUM_TRIALS
  p = argparse.ArgumentParser()
  p.add_argument('--ga-csv', required=True)
  p.add_argument('--states-files', nargs='+', required=True)
  p.add_argument('--m', type=int, default=8)
  p.add_argument('--generations', type=int, nargs='+',
                 default=[0, 2, 5, 10, 15, 20, 25, 29])
  p.add_argument('--num-trials', type=int, default=10)
  p.add_argument('--workers', type=int, default=16)
  p.add_argument('--output-csv', required=True)
  args = p.parse_args()
  NUM_TRIALS = args.num_trials

  ga = pd.read_csv(args.ga_csv)
  ga = ga[ga.max_num_features == args.m]
  tasks = []
  for net, g in ga.groupby('original_network_idx'):
    have = sorted(g.generation.unique())
    picks = sorted({min(have, key=lambda h: abs(h - t))
                    for t in args.generations})
    rows = g.set_index('generation')
    gens = [(int(gen), float(rows.loc[gen].best_accuracy),
             sorted(ast.literal_eval(rows.loc[gen].features)))
            for gen in picks]
    tasks.append((int(net), gens))
  print(f'{len(tasks)} networks, {sum(len(g) for _, g in tasks)} panels')

  _load(args.states_files)
  with multiprocessing.Pool(args.workers) as pool:
    all_rows = [r for rows in pool.imap_unordered(_score_network, tasks)
                for r in rows]
  out = pathlib.Path(args.output_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(all_rows).sort_values(
    ['original_network_idx', 'generation']).to_csv(out, index=False)
  print(f'wrote {out} ({len(all_rows)} rows)')


if __name__ == '__main__':
  main()
