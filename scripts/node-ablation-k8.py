'''Node-ablation study on the GA-evolved k=8 sensor sets.

For each network's evolved exactly-k=8 set we retrain the SAME RandomForest
evaluator the GA used (classifier.train_and_test) after removing m = 1, 2, 3
nodes, and record the accuracy drop against how many of the removed nodes were
'sensitive' (per-node Hamming distance from control B > antimode cutoff).

This is the 'backwards' knockout view: it asks whether the accuracy actually
*depends* on the sensitive nodes, which is more informative than forward
selection since many k-subsets can classify well.

Reuses:
  - scripts/classifier.py  -> train_and_test / get_score (exact GA fitness).
  - data/sensitivity/B-rho{rho}.npz  -> per-node sensitivity B (from the
    sensitive-nodes-vs-rho notebook, Section 1).

Run per rho, e.g.:
  python scripts/node-ablation-k8.py --rho 0.5
  python scripts/node-ablation-k8.py --rho 0.99 --n-trials 15

Output: data/sensitivity/ablation-k8-rho{rho}.csv
'''
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import train_and_test

K = 8

# rho -> paths (relative to repo root; run from the repo root).
DATASETS = {
  0.5: {
    'states': 'data/drug-fixed-targets-v7/N5000/derived/states-1772488362007.csv',
    'ga': 'data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv',
    'b': 'data/sensitivity/B-rho0.5.npz',
  },
  0.99: {
    'states': 'data/drug-fixed-targets-v5/N5000/derived/states-1771990942417.csv',
    'ga': 'data/drug-fixed-targets-v5/N5000/ga-results-v5/combined-full.csv',
    'b': 'data/sensitivity/B-rho0.99.npz',
  },
}


def load_ga_subsets(ga_csv, k):
  '''network idx -> node indices for the final-generation best size-k set.'''
  ga = pd.read_csv(ga_csv)
  ga = ga[ga['max_num_features'] == k]
  final_idx = ga.groupby('original_network_idx')['generation'].idxmax()
  out = {}
  for _, r in ga.loc[final_idx].iterrows():
    nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
    out[int(r['original_network_idx'])] = nodes
  return out


def antimode_cutoff(B, lo=0.05, hi=0.40):
  '''Valley of the pooled B distribution between the insensitive spike and tail.'''
  bins = np.linspace(0, 1, 101)
  counts, edges = np.histogram(B.ravel(), bins=bins)
  centers = 0.5 * (edges[:-1] + edges[1:])
  smooth = np.convolve(counts, np.ones(5) / 5, mode='same')
  window = (centers > lo) & (centers < hi)
  return float(centers[window][np.argmin(smooth[window])])


def score(net_df, node_idxs, network_idx, n_trials):
  '''Mean RF plurality-vote accuracy over n_trials splits (GA's get_score).'''
  dep_vars = [f'node-{n}' for n in node_idxs]
  perf, _ = train_and_test(
    net_df, num_trials=n_trials, original_network_idx=network_idx, dep_vars=dep_vars,
  )
  return float(perf['correct'].mean())


def main(args):
  paths = DATASETS[args.rho] if args.rho in DATASETS else None
  states_file = args.states_file or paths['states']
  ga_file = args.ga_file or paths['ga']
  b_file = args.b_file or paths['b']

  subsets = load_ga_subsets(ga_file, K)
  # Keep only sets that are exactly K distinct nodes.
  exact = {net: nodes for net, nodes in subsets.items() if len(set(nodes)) == K}
  if args.networks:
    lo, hi = (int(x) for x in args.networks.split('-'))
    exact = {net: nodes for net, nodes in exact.items() if lo <= net <= hi}
  dropped = sorted(set(subsets) - set(exact))
  print(f'rho={args.rho}: {len(exact)}/{len(subsets)} networks have exactly {K} nodes', flush=True)
  if dropped:
    print(f'  skipped (size != {K}): {dropped}', flush=True)

  b_data = np.load(b_file)
  B = b_data['B']
  b_nets = list(int(x) for x in b_data['networks'])
  cutoff = args.cutoff if args.cutoff is not None else antimode_cutoff(B)
  print(f'  sensitivity cutoff (antimode) = {cutoff:.4f}', flush=True)

  # Read only the node columns we need (union across networks) + metadata.
  needed = sorted({n for nodes in exact.values() for n in nodes})
  usecols = ['drug_name', 'original_network_idx', 'initial_condition_idx', 'step_num']
  usecols += [f'node-{n}' for n in needed]
  print(f'  reading {len(usecols)} columns from {states_file} ...', flush=True)
  df = pd.read_csv(states_file, usecols=usecols).rename(columns={'drug_name': 'Drug'})

  rows = []
  for i, (net, nodes) in enumerate(sorted(exact.items())):
    net_df = df[df['original_network_idx'] == net]
    b_row = B[b_nets.index(net)]
    is_sensitive = {n: bool(b_row[n] > cutoff) for n in nodes}
    baseline = score(net_df, nodes, net, args.n_trials)
    for m in range(1, args.max_remove + 1):
      for removed in itertools.combinations(nodes, m):
        remaining = [n for n in nodes if n not in removed]
        ablated = score(net_df, remaining, net, args.n_trials)
        rows.append({
          'rho': args.rho,
          'original_network_idx': net,
          'm_removed': m,
          'removed_nodes': list(removed),
          'n_sensitive_removed': int(sum(is_sensitive[n] for n in removed)),
          'meanB_removed': float(np.mean([b_row[n] for n in removed])),
          'baseline_acc': baseline,
          'ablated_acc': ablated,
          'acc_drop': baseline - ablated,
        })
    print(f'  [{i + 1}/{len(exact)}] net {net}: baseline={baseline:.3f}, '
          f'sensitive={sum(is_sensitive.values())}/{K}', flush=True)

  out = pd.DataFrame(rows)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  out.to_csv(args.out, index=False)
  print(f'saved {args.out}  shape={out.shape}', flush=True)
  print(out.groupby(['m_removed', 'n_sensitive_removed'])['acc_drop'].mean().round(3))


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--rho', type=float, required=True, help='0.5 or 0.99 (uses built-in paths)')
  p.add_argument('--n-trials', type=int, default=10, help='RF train/test splits averaged per subset')
  p.add_argument('--max-remove', type=int, default=3, help='remove m = 1..max-remove nodes')
  p.add_argument('--cutoff', type=float, default=None, help='sensitivity cutoff (default: antimode of B)')
  p.add_argument('--states-file', type=str, default=None)
  p.add_argument('--ga-file', type=str, default=None)
  p.add_argument('--b-file', type=str, default=None)
  p.add_argument('--out', type=str, default=None)
  p.add_argument('--networks', type=str, default=None,
                 help='restrict to an inclusive network index range, e.g. 0-4')
  args = p.parse_args()
  if args.out is None:
    args.out = f'data/sensitivity/ablation-k8-rho{args.rho}.csv'
  return args


if __name__ == '__main__':
  main(parse_args())
