#!/usr/bin/env python3
'''Is a reporter's answer reproducible from trial to trial?

Every deviation Delta_{j,q} in this paper is an average over replicates, so
two nodes with the same average can differ in how often an individual trial
reproduces it. That difference is invisible to any statistic computed from
the mean response profile, and the panel comparison shows such statistics do
not order panels by accuracy. This script measures the missing quantity.

For each node j and shock q, using the retained snapshots of every replicate:

  mu[j,q]   mean state of node j under shock q, across replicates and snapshots
  sd[j,q]   standard deviation of that state across replicates

The discriminability of two shocks at node j is then the standardised
difference

  dprime[j,q,q'] = |mu[j,q] - mu[j,q']| / sqrt((sd[j,q]^2 + sd[j,q']^2) / 2),

the usual signal to noise ratio for telling two conditions apart from single
trials. A node with a large mean difference but a large trial to trial spread
has a small d prime and is a poor reporter despite looking good in the mean
profile.

Output is one row per node with its class, its mean based separation, and its
reliability based separation, so the two can be compared directly.

Usage:
  python scripts/reporter-reliability.py \
    --original-network-idx 0 \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --b-file data/sensitivity/B-rho0.5.npz \
    --output-dir data/reliability/rho0.5
'''
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

SPLIT = 6


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--original-network-idx', type=int, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--s-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--output-dir', type=str, required=True)
  args = p.parse_args()

  out = pathlib.Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  done = out / f'{args.original_network_idx}-full.done'
  if done.exists():
    return

  net = args.original_network_idx
  sd_ = np.load(args.s_file, allow_pickle=True)
  S = sd_['S'].transpose(0, 2, 1)                       # (nets, nodes, shocks)
  snets = [int(x) for x in sd_['networks']]
  bd = np.load(args.b_file)
  B, bnets = bd['B'], [int(x) for x in bd['networks']]
  cut = antimode(B)
  si, bi = snets.index(net), bnets.index(net)
  n_row = (S[si] >= cut).sum(axis=1)

  df = pd.read_csv(args.states_file, index_col=0).reset_index()
  df = df.rename(columns={'drug_name': 'Drug'})
  df = df[df.original_network_idx == net]
  node_cols = [c for c in df.columns if c.startswith('node-')]
  idx = np.array([int(c.split('-')[1]) for c in node_cols])

  # per shock mean and across-replicate spread, one value per node
  drugs = sorted(df['Drug'].unique())
  mu = np.zeros((len(drugs), len(node_cols)))
  sd = np.zeros((len(drugs), len(node_cols)))
  for k, dr in enumerate(drugs):
    sub = df[df['Drug'] == dr]
    # average within a replicate first, so the spread is across replicates
    per_rep = sub.groupby('initial_condition_idx')[node_cols].mean().to_numpy()
    mu[k] = per_rep.mean(axis=0)
    sd[k] = per_rep.std(axis=0, ddof=1)

  pairs = list(itertools.combinations(range(len(drugs)), 2))
  pooled = np.sqrt((sd[[a for a, _ in pairs]] ** 2 + sd[[b for _, b in pairs]] ** 2) / 2)
  diff = np.abs(mu[[a for a, _ in pairs]] - mu[[b for _, b in pairs]])
  with np.errstate(divide='ignore', invalid='ignore'):
    dprime = np.where(pooled > 1e-9, diff / pooled, 0.0)

  np.savez_compressed(out / f'{net}-pairs.npz',
                      mu=mu.astype(np.float32), sd=sd.astype(np.float32),
                      dprime=dprime.astype(np.float32),
                      drugs=np.array(drugs, dtype=object), nodes=idx)

  cls = np.where(n_row[idx] == 0, 'unresponsive',
                 np.where(n_row[idx] >= SPLIT, 'promiscuous', 'dormant'))
  rows = pd.DataFrame(dict(
      original_network_idx=net, node=idx, cls=cls,
      S=B[bi, idx],
      mean_sep=diff.mean(axis=0),          # mean based separation, what the profile shows
      max_mean_sep=diff.max(axis=0),
      spread=sd.mean(axis=0),              # trial to trial spread
      dprime=dprime.mean(axis=0),          # reliability based separation
      max_dprime=dprime.max(axis=0),
      theta=cut))
  rows.to_csv(out / f'{net}-full.csv', index=False)
  done.touch()
  g = rows.groupby('cls')[['mean_sep', 'spread', 'dprime']].mean()
  print(g.to_string(), flush=True)


if __name__ == '__main__':
  main()
