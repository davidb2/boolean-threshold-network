'''Per-shock damage from removing one reporter from an evolved panel.

The deep ablation records only overall accuracy. This script records the
per-shock recall instead, so that the SHAPE of the damage can be compared
between reporter classes: removing a broadly responding sensitive member
should degrade every shock a little, while removing an insensitive member
that switches coherently under one or two shocks should degrade those
shocks and leave the rest intact.

For each network's evolved k = 8 panel we retrain the same RandomForest
evaluator the genetic algorithm used, once for the intact panel and once
for each single member removed, and write the per-shock recall in both
conditions.

Output columns:
  rho, original_network_idx, removed_node, is_sensitive, B_removed,
  shock, recall_full, recall_ablated, recall_drop

Run per rho, e.g.:
  python scripts/per-shock-ablation.py --rho 0.5 --n-trials 30
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import train_and_test

K = 8


def load_ga_subsets(ga_csv, k):
  ga = pd.read_csv(ga_csv)
  ga = ga[ga['max_num_features'] == k]
  final_idx = ga.groupby('original_network_idx')['generation'].idxmax()
  out = {}
  for _, r in ga.loc[final_idx].iterrows():
    nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
    out[int(r['original_network_idx'])] = nodes
  return out


def antimode_cutoff(B, lo=0.05, hi=0.40):
  bins = np.linspace(0, 1, 101)
  counts, edges = np.histogram(B.ravel(), bins=bins)
  centers = 0.5 * (edges[:-1] + edges[1:])
  smooth = np.convolve(counts, np.ones(5) / 5, mode='same')
  window = (centers > lo) & (centers < hi)
  return float(centers[window][np.argmin(smooth[window])])


def per_shock_recall(net_df, node_idxs, network_idx, n_trials):
  '''Recall per shock, from the same evaluator the GA used.'''
  dep_vars = [f'node-{n}' for n in node_idxs]
  perf, _ = train_and_test(
    net_df, num_trials=n_trials, original_network_idx=network_idx, dep_vars=dep_vars,
  )
  return perf.groupby('drug_actual')['correct'].mean()


def main(args):
  subsets = load_ga_subsets(args.ga_file, K)
  exact = {net: nodes for net, nodes in subsets.items() if len(set(nodes)) == K}
  if args.networks:
    lo, hi = (int(x) for x in args.networks.split('-'))
    exact = {net: nodes for net, nodes in exact.items() if lo <= net <= hi}
  print(f'rho={args.rho}: {len(exact)} networks with exactly {K} nodes', flush=True)

  b_data = np.load(args.b_file)
  B = b_data['B']
  b_nets = [int(x) for x in b_data['networks']]
  cutoff = args.cutoff if args.cutoff is not None else antimode_cutoff(B)
  print(f'  antimode cutoff = {cutoff:.4f}', flush=True)

  needed = sorted({n for nodes in exact.values() for n in nodes})
  usecols = ['drug_name', 'original_network_idx', 'initial_condition_idx', 'step_num']
  usecols += [f'node-{n}' for n in needed]
  print(f'  reading {len(usecols)} columns from {args.states_file} ...', flush=True)
  df = pd.read_csv(args.states_file, usecols=usecols).rename(columns={'drug_name': 'Drug'})

  rows = []
  for i, (net, nodes) in enumerate(sorted(exact.items())):
    net_df = df[df['original_network_idx'] == net]
    b_row = B[b_nets.index(net)]
    full = per_shock_recall(net_df, nodes, net, args.n_trials)
    for removed in nodes:
      remaining = [n for n in nodes if n != removed]
      abl = per_shock_recall(net_df, remaining, net, args.n_trials)
      for shock in full.index:
        rows.append(dict(
          rho=args.rho, original_network_idx=net, removed_node=removed,
          is_sensitive=bool(b_row[removed] > cutoff), B_removed=float(b_row[removed]),
          shock=shock, recall_full=float(full[shock]),
          recall_ablated=float(abl.get(shock, np.nan)),
          recall_drop=float(full[shock] - abl.get(shock, np.nan)),
        ))
    print(f'  [{i + 1}/{len(exact)}] network {net} done', flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
  print(f'wrote {args.out} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--rho', type=str, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--n-trials', type=int, default=30)
  p.add_argument('--cutoff', type=float, default=None)
  p.add_argument('--networks', type=str, default=None)
  p.add_argument('--out', type=str, required=True)
  main(p.parse_args())
