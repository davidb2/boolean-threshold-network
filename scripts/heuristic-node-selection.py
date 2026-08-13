#!/usr/bin/env python3
'''Per-network heuristic node selection baselines.

Implements five selection strategies to compare against the genetic algorithm
and random selection, all scored with the same random forest evaluator
(classifier.train_and_test):

  sensitivity  top nodes by per-node Hamming distance from control (needs --b-file)
  in-degree    most regulated nodes (needs --networks-file)
  out-degree   biggest hubs (needs --networks-file)
  mmse         greedy covariance column selection on control snapshots
  jaccard      greedy coverage of distinct downstream influence sets (needs --networks-file)

Each strategy produces a ranking; prefixes of the ranking are evaluated at each
feature size. Output matches random-node-selection.py ({i}-full.csv + .done).

Usage:
  python scripts/heuristic-node-selection.py \
    --strategy sensitivity \
    --original-network-idx 0 \
    --states-file data/drug-fixed-targets-v5/N5000/derived/states-*.csv \
    --b-file data/sensitivity/B-rho0.99.npz \
    --network-size 5000 \
    --feature-sizes 1 2 4 8 16 32 64 128 \
    --output-dir data/selection-strategies/rho0.99/sensitivity-results
'''
import argparse
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from typing import *

from classifier import train_and_test


def load_edges(networks_file, network_idx):
  df = pd.read_csv(networks_file)
  df = df[df['original_network_idx'] == network_idx]
  return df[['source', 'target']].to_numpy(dtype=np.int64)


def rank_by_degree(edges, n, direction, rng):
  col = 0 if direction == 'out' else 1
  deg = np.bincount(edges[:, col], minlength=n)
  jitter = rng.random(n)
  return np.lexsort((jitter, -deg))


def rank_by_sensitivity(b_file, network_idx, n, rng):
  data = np.load(b_file)
  nets = [int(x) for x in data['networks']]
  b = data['B'][nets.index(network_idx)]
  assert len(b) == n
  jitter = rng.random(n)
  return np.lexsort((jitter, -b))


def rank_by_mmse(states_df, node_cols, k_max, ridge=1e-9, tol=1e-12):
  ctrl = states_df[states_df['Drug'] == 'control'][node_cols]
  X = ctrl.to_numpy(dtype=np.float64)
  X = X - X.mean(axis=0)
  sigma = (X.T @ X) / (X.shape[0] - 1)
  n = sigma.shape[0]
  sigma = sigma + ridge * np.eye(n)

  S = []
  selected = np.zeros(n, dtype=bool)
  diag = np.diag(sigma).copy()
  for _ in range(k_max):
    if not S:
      s_all = diag
      qnorm = (sigma ** 2).sum(axis=0)
    else:
      sigma_s = sigma[:, S]
      ainv = np.linalg.inv(sigma[np.ix_(S, S)])
      v = ainv @ sigma_s.T
      s_all = diag - np.einsum('ij,ji->i', sigma_s, v)
      resid = sigma - sigma_s @ v
      qnorm = (resid ** 2).sum(axis=0)
    gain = np.where((s_all > tol) & ~selected, qnorm / np.maximum(s_all, tol), -np.inf)
    j = int(np.argmax(gain))
    if not np.isfinite(gain[j]):
      break
    S.append(j)
    selected[j] = True
  return np.array(S, dtype=np.int64)


def rank_by_jaccard(edges, n, k_max, rng):
  succ = {}
  for s, t in edges:
    succ.setdefault(int(s), set()).add(int(t))
  sizes = {v: len(nb) for v, nb in succ.items()}
  covered = set()
  remaining = set(range(n))
  order = []
  jitter = rng.random(n)
  for _ in range(min(k_max, n)):
    best, best_key = None, None
    ca = len(covered)
    for v in remaining:
      nb = succ.get(v)
      if nb is None:
        dis = 0.0 if ca == 0 else 1.0
      else:
        inter = len(covered & nb)
        union = ca + sizes[v] - inter
        dis = 1.0 - (inter / union) if union else 0.0
      key = (dis, jitter[v])
      if best_key is None or key > best_key:
        best, best_key = v, key
    order.append(best)
    remaining.discard(best)
    if best in succ:
      covered |= succ[best]
  return np.array(order, dtype=np.int64)


def _score_one(task):
  features, network_idx, df = task
  perf, _ = train_and_test(
    df, num_trials=1, original_network_idx=network_idx, dep_vars=list(features),
  )
  return float(perf['correct'].mean())


def get_ranking(args, states_df, node_cols, k_max, rng):
  if args.strategy == 'sensitivity':
    return rank_by_sensitivity(args.b_file, args.original_network_idx, args.network_size, rng)
  if args.strategy in ('in-degree', 'out-degree'):
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_degree(edges, args.network_size, args.strategy.split('-')[0], rng)
  if args.strategy == 'mmse':
    return rank_by_mmse(states_df, node_cols, k_max)
  if args.strategy == 'jaccard':
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_jaccard(edges, args.network_size, k_max, rng)
  raise ValueError(f'unknown strategy {args.strategy}')


def main(args):
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  output_path = out_dir / f'{args.original_network_idx}-full.csv'
  done_path = out_dir / f'{args.original_network_idx}-full.done'
  if done_path.exists():
    return

  states_df = pd.read_csv(args.states_file, index_col=0)
  states_df = states_df.reset_index().rename(columns={'drug_name': 'Drug'})
  states_df = states_df[states_df['original_network_idx'] == args.original_network_idx]
  states_df = states_df.drop(columns=['original_network_idx', 'initial_condition_idx'])
  node_cols = [f'node-{i}' for i in range(args.network_size)]

  k_max = max(args.feature_sizes)
  rows = []
  tasks = []
  for trial in range(args.num_trials):
    rng = np.random.default_rng(args.seed + trial)
    ranking = get_ranking(args, states_df, node_cols, k_max, rng)
    for k in args.feature_sizes:
      if k > len(ranking):
        print(f'skipping k={k}: ranking only has {len(ranking)} nodes', flush=True)
        continue
      features = [f'node-{i}' for i in ranking[:k]]
      tasks.append((trial, k, features))

  with multiprocessing.Pool(processes=args.num_workers) as pool:
    accs = pool.map(
      _score_one,
      [(features, args.original_network_idx, states_df) for _, _, features in tasks],
    )

  for (trial, k, features), acc in zip(tasks, accs):
    rows.append({
      'original_network_idx': args.original_network_idx,
      'strategy': args.strategy,
      'max_num_features': k,
      'trial': trial,
      'accuracy': acc,
      'features': sorted(features),
    })
    print(f'k={k}, trial={trial}, accuracy={acc:.4f}', flush=True)

  output_path.unlink(missing_ok=True)
  pd.DataFrame(rows).to_csv(output_path, index=False)
  done_path.touch()


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--strategy', type=str, required=True,
                 choices=['sensitivity', 'in-degree', 'out-degree', 'mmse', 'jaccard'])
  p.add_argument('--original-network-idx', type=int, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--networks-file', type=str, default=None)
  p.add_argument('--b-file', type=str, default=None)
  p.add_argument('--output-dir', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--feature-sizes', type=int, nargs='+', required=True)
  p.add_argument('--num-trials', type=int, default=1)
  p.add_argument('--num-workers', type=int, default=4)
  p.add_argument('--seed', type=int, default=2025)
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
