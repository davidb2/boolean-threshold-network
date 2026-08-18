#!/usr/bin/env python3
'''Why do evolved panels keep promiscuous reporters at all?

A dormant reporter is a test that fires on a few shocks, so a panel of eight
of them looks like it should be an ideal separating design: eight binary
tests can in principle label far more than eleven alternatives. This script
builds that panel explicitly and scores it against the alternatives with the
same classifier used everywhere else, so the question is settled by
measurement rather than by argument.

Panels compared, all at m = 8 on one network:

  greedy-cover      dormant nodes chosen greedily to cover as many distinct
                    shocks as possible, the natural group testing design
  greedy-margin     dormant nodes chosen greedily to maximise the WORST CASE
                    separation between any two alternatives, which is the
                    quantity a classifier is actually limited by
  promiscuous       the eight nodes of highest mean sensitivity, that is the
                    responsiveness heuristic
  dormant-random    eight dormant nodes drawn at random, the control for
                    greedy-cover
  evolved           the genetic algorithm panel, for reference
  half-and-half     four promiscuous plus four greedy-margin dormant nodes

The separation of two alternatives on a panel V is the L1 distance between
their mean deviation vectors restricted to V, where the control is the zero
vector. The margin of a panel is the minimum of that distance over all
pairs of the d+1 alternatives.

Usage:
  python scripts/panel-design-comparison.py \
    --original-network-idx 0 \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --b-file data/sensitivity/B-rho0.5.npz \
    --ga-file data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
    --network-size 5000 --num-trials 10 \
    --output-dir data/panel-design/rho0.5
'''
import argparse
import ast
import itertools
import pathlib

import numpy as np
import pandas as pd

from classifier import train_and_test

M = 8
SPLIT = 6          # answering this many shocks or more makes a node promiscuous


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def margin(Sv):
  '''Worst case L1 separation between any two of the d+1 alternatives.

  Sv is (nodes, shocks). The control contributes the zero vector, so a
  shock's separation from control is just the L1 norm of its own column.
  '''
  d = Sv.shape[1]
  best = [float(np.abs(Sv[:, q] - Sv[:, r]).sum())
          for q, r in itertools.combinations(range(d), 2)]
  best += [float(np.abs(Sv[:, q]).sum()) for q in range(d)]
  return float(np.min(best)), float(np.mean(best))


def greedy_cover(A_row, pool):
  '''Dormant nodes chosen to cover as many distinct shocks as possible.'''
  pool, chosen, covered = list(pool), [], np.zeros(A_row.shape[1], bool)
  for _ in range(M):
    gain = (A_row[pool] & ~covered).sum(axis=1)
    j = int(np.argmax(gain))
    chosen.append(pool[j])
    covered |= A_row[pool[j]]
    pool.pop(j)
  return chosen


def greedy_margin(S_row, pool, seed):
  '''Nodes chosen greedily to maximise the worst case separation.

  Starting from the single node with the largest margin, each step adds the
  candidate that most improves the minimum pairwise separation. The pool is
  subsampled for tractability, since every candidate is scored at every step.
  '''
  rng = np.random.default_rng(seed)
  pool = list(rng.choice(pool, size=min(400, len(pool)), replace=False))
  chosen = []
  for _ in range(M):
    best, best_j = -np.inf, None
    for j in pool:
      mn, _ = margin(S_row[chosen + [j]])
      if mn > best:
        best, best_j = mn, j
    chosen.append(best_j)
    pool.remove(best_j)
  return chosen


def score(states_df, net_idx, nodes, num_trials):
  feats = [f'node-{i}' for i in nodes]
  perf, _ = train_and_test(states_df, num_trials=num_trials,
                           original_network_idx=net_idx, dep_vars=feats)
  return float(perf['correct'].mean())


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--original-network-idx', type=int, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--s-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--num-trials', type=int, default=10)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--output-dir', type=str, required=True)
  args = p.parse_args()

  out = pathlib.Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  done = out / f'{args.original_network_idx}-full.done'
  if done.exists():
    return

  sd = np.load(args.s_file, allow_pickle=True)
  S = sd['S'].transpose(0, 2, 1)                 # (nets, nodes, shocks)
  snets = [int(x) for x in sd['networks']]
  bd = np.load(args.b_file)
  B, bnets = bd['B'], [int(x) for x in bd['networks']]
  cut = antimode(B)
  net = args.original_network_idx
  si, bi = snets.index(net), bnets.index(net)
  S_row, B_row = S[si], B[bi]
  A_row = S_row >= cut
  n_row = A_row.sum(axis=1)

  dormant = np.where((n_row >= 1) & (n_row <= SPLIT - 1))[0]
  promisc = np.where(n_row >= SPLIT)[0]
  rng = np.random.default_rng(args.seed)

  ga = pd.read_csv(args.ga_file)
  ga = ga[ga.max_num_features == M]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  row = fin[fin.original_network_idx == net]
  evolved = ([int(s.split('-')[1]) for s in ast.literal_eval(row['features'].iloc[0])]
             if len(row) else None)

  cover = greedy_cover(A_row, dormant)
  marg = greedy_margin(S_row, dormant, args.seed)
  top8 = list(np.argsort(-B_row)[:M])
  dorm_rand = list(rng.choice(dormant, M, replace=False))
  mixed = list(np.argsort(-B_row[promisc])[:4].astype(int))
  mixed = [int(promisc[k]) for k in mixed] + marg[:4]

  panels = {'greedy-cover': cover, 'greedy-margin': marg, 'promiscuous': top8,
            'dormant-random': dorm_rand, 'half-and-half': mixed}
  if evolved is not None:
    panels['evolved'] = evolved

  states_df = pd.read_csv(args.states_file, index_col=0)
  states_df = states_df.reset_index().rename(columns={'drug_name': 'Drug'})
  states_df = states_df[states_df.original_network_idx == net]
  states_df = states_df.drop(columns=['original_network_idx', 'initial_condition_idx'])

  rows = []
  for name, nodes in panels.items():
    mn, mean = margin(S_row[nodes])
    acc = score(states_df, net, nodes, args.num_trials)
    n_dorm = int(((n_row[nodes] >= 1) & (n_row[nodes] <= SPLIT - 1)).sum())
    n_prom = int((n_row[nodes] >= SPLIT).sum())
    covered = int(A_row[nodes].any(axis=0).sum())
    patterns = len({tuple(A_row[nodes][:, q]) for q in range(A_row.shape[1])})
    rows.append(dict(original_network_idx=net, panel=name, accuracy=acc,
                     min_margin=mn, mean_margin=mean, shocks_covered=covered,
                     distinct_patterns=patterns, n_dormant=n_dorm,
                     n_promiscuous=n_prom, theta=cut,
                     nodes=' '.join(map(str, nodes))))
    print(f'{name:16s} acc {acc:.4f}  margin {mn:.3f}  covered {covered}/10  '
          f'patterns {patterns}/10  {n_prom}P/{n_dorm}D', flush=True)

  pd.DataFrame(rows).to_csv(out / f'{net}-full.csv', index=False)
  done.touch()


if __name__ == '__main__':
  main()
