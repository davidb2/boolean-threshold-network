#!/usr/bin/env python3
'''The set level rule as a ranking, its prefixes scored at every size.

The greedy sequence is nested, so one pass to the largest size gives the
panel at every smaller size. Each prefix is scored on fresh random splits
of the standard evaluator, the same protocol as every other strategy.
The sequence stops at 32 members, where the shrunk covariance the rule
inverts is still well conditioned given roughly one hundred replicates.

Usage:
  python scripts/rule-prefix-curves.py \
    --states-files ... --pairs-dir data/reliability-pairs/rho0.5 \
    --sizes 1 2 4 8 16 32 --num-trials 10 --workers 16 \
    --output-csv ...
'''
import argparse
import itertools
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from classifier import train_and_test

STATES = None
ARGS = None


def _load(states_files):
  global STATES
  frames = []
  for f in states_files:
    df = pd.read_csv(f, index_col=0).reset_index()
    df = df.rename(columns={'drug_name': 'Drug'})
    frames.append(df)
  STATES = pd.concat(frames, ignore_index=True)
  print(f'loaded {len(STATES)} state rows', flush=True)


def _one(net):
  pairs_file = pathlib.Path(ARGS.pairs_dir) / f'{net}-pairs.npz'
  if not pairs_file.exists():
    print(f'net {net}: no pairs file, skipped', flush=True)
    return []
  z = np.load(pairs_file, allow_pickle=True)
  DP = z['dprime'].astype(float)
  node_ids = z['nodes']
  mean_dp = DP.mean(axis=0)
  pool_nodes = [int(node_ids[i]) for i in np.argsort(-mean_dp)[:400]]

  sub = STATES[STATES.original_network_idx == net]
  cols = [f'node-{i}' for i in pool_nodes]
  drugs = sorted(sub['Drug'].unique())
  reads = {d: sub[sub['Drug'] == d]
               .groupby('initial_condition_idx')[cols].mean().to_numpy()
           for d in drugs}
  mus = {d: v.mean(axis=0) for d, v in reads.items()}
  centered = np.vstack([v - v.mean(axis=0) for v in reads.values()])
  dof = max(len(centered) - len(drugs), 1)
  pairs_dr = list(itertools.combinations(drugs, 2))

  chosen, remaining = [], list(range(len(pool_nodes)))
  for _ in range(max(ARGS.sizes)):
    best, bj = -np.inf, None
    for j in remaining:
      idxs = chosen + [j]
      C = centered[:, idxs]
      Sg = C.T @ C / dof
      Sg = 0.5 * Sg + 0.5 * np.diag(np.diag(Sg)) + 1e-6 * np.eye(len(idxs))
      Si = np.linalg.inv(Sg)
      vals = [float(np.sqrt(max((mus[a][idxs] - mus[b][idxs]) @ Si
                                @ (mus[a][idxs] - mus[b][idxs]), 0.0)))
              for a, b in pairs_dr]
      v = np.percentile(vals, 10)
      if v > best:
        best, bj = v, j
    chosen.append(bj)
    remaining.remove(bj)
  seq = [pool_nodes[j] for j in chosen]
  if ARGS.sequences_only:
    print(f'net {net} sequence done', flush=True)
    return [dict(original_network_idx=net, step=k + 1, node=n)
            for k, n in enumerate(seq)]

  drop = [c for c in ('original_network_idx', 'initial_condition_idx')
          if c in sub.columns]
  eval_df = sub.drop(columns=drop)
  rows = []
  for m in ARGS.sizes:
    feats = sorted(f'node-{i}' for i in seq[:m])
    perf, _ = train_and_test(eval_df, num_trials=ARGS.num_trials,
                             original_network_idx=net, dep_vars=feats)
    acc = float(perf['correct'].mean())
    rows.append(dict(original_network_idx=net, max_num_features=m,
                     accuracy=acc))
    print(f'net {net} m {m}: {acc:.3f}', flush=True)
  return rows


def main():
  global ARGS
  p = argparse.ArgumentParser()
  p.add_argument('--states-files', nargs='+', required=True)
  p.add_argument('--pairs-dir', required=True)
  p.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
  p.add_argument('--num-trials', type=int, default=10)
  p.add_argument('--workers', type=int, default=16)
  p.add_argument('--output-csv', required=True)
  p.add_argument('--sequences-only', action='store_true',
                 help='write the greedy pick order (net, step, node) and skip evaluation')
  ARGS = p.parse_args()

  _load(ARGS.states_files)
  nets = sorted(int(x) for x in STATES.original_network_idx.unique())
  print(f'{len(nets)} networks', flush=True)
  with multiprocessing.Pool(ARGS.workers) as pool:
    all_rows = [r for rows in pool.imap_unordered(_one, nets) for r in rows]
  out = pathlib.Path(ARGS.output_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  keys = ['original_network_idx',
          'step' if ARGS.sequences_only else 'max_num_features']
  pd.DataFrame(all_rows).sort_values(keys).to_csv(out, index=False)
  print(f'wrote {out} ({len(all_rows)} rows)')


if __name__ == '__main__':
  main()
