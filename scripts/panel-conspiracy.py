#!/usr/bin/env python3
'''Which member pairs carry the covariance advantage?

The covariance aware discriminability orders the panel designs and the
independence assuming one does not, so the advantage of the evolved panels
lives in the off diagonal of the member covariance. This script asks which
off diagonal entries: the ones between two promiscuous members, between two
dormant members, or between a promiscuous and a dormant member. If the
cross class block carries the gain, the two classes are literally the
conspirators.

Attribution is exact. The off diagonal of the shrunk covariance is split
into three blocks by member pair type, PP, PD, and DD (pairs involving an
unresponsive member join a residual block, UX). The value of a block
coalition is the tenth percentile over all condition pairs of the
Mahalanobis discriminability computed with only those blocks present, the
diagonal always kept, and each block's contribution is its Shapley value
over the 2^4 coalitions. The contributions sum exactly to the full minus
diagonal only difference. Covariances are floored at a small positive
eigenvalue when a coalition breaks positive definiteness.

A second, sign rule statistic is reported per block: the fraction of
(member pair, condition pair) combinations, weighted by |r|, in which the
noise correlation opposes the signal product, which is the geometry that
lets shared noise cancel.

Usage:
  python scripts/panel-conspiracy.py \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --panels-csv data/panel-design/rho0.5/combined.csv \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --b-file data/sensitivity/B-rho0.5.npz \
    --output-csv data/panel-design/rho0.5/conspiracy.csv
'''
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

SPLIT = 6
BLOCKS = ['PP', 'PD', 'DD', 'UX']
LAM = 0.5


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def psd(M):
  w, V = np.linalg.eigh(M)
  return (V * np.maximum(w, 1e-8)) @ V.T


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--states-file', required=True)
  p.add_argument('--panels-csv', required=True)
  p.add_argument('--s-file', required=True)
  p.add_argument('--b-file', required=True)
  p.add_argument('--output-csv', required=True)
  args = p.parse_args()

  sd_ = np.load(args.s_file, allow_pickle=True)
  S_all = sd_['S'].transpose(0, 2, 1)
  snets = [int(x) for x in sd_['networks']]
  bd = np.load(args.b_file)
  B_all = bd['B']
  cut = antimode(B_all)

  panels = pd.read_csv(args.panels_csv)
  df = pd.read_csv(args.states_file, index_col=0).reset_index()
  df = df.rename(columns={'drug_name': 'Drug'})

  from math import factorial
  nB = len(BLOCKS)
  weights = {k: factorial(k) * factorial(nB - k - 1) / factorial(nB) for k in range(nB)}

  rows = []
  for net, g in panels.groupby('original_network_idx'):
    si = snets.index(int(net))
    n_row = (S_all[si] >= cut).sum(axis=1)
    sub = df[df.original_network_idx == net]
    drugs = sorted(sub['Drug'].unique())
    cond_pairs = list(itertools.combinations(range(len(drugs)), 2))
    for _, r in g.iterrows():
      nodes = [int(x) for x in r['nodes'].split()]
      members = [f'node-{x}' for x in nodes]
      m = len(members)
      cls = ['U' if n_row[x] == 0 else 'P' if n_row[x] >= SPLIT else 'D' for x in nodes]

      def ptype(i, j):
        a, b = sorted((cls[i], cls[j]))
        if 'U' in (a, b):
          return 'UX'
        return {('P', 'P'): 'PP', ('D', 'P'): 'PD', ('D', 'D'): 'DD'}[(a, b)]

      reads = {d: sub[sub['Drug'] == d].groupby('initial_condition_idx')[members]
                   .mean().to_numpy() for d in drugs}
      mus = np.array([v.mean(axis=0) for v in reads.values()])
      centered = np.vstack([v - v.mean(axis=0) for v in reads.values()])
      S = centered.T @ centered / max(len(centered) - len(drugs), 1)
      diag = np.diag(np.diag(S))
      off = (1 - LAM) * (S - diag)          # match the shrinkage used before

      block_mask = {b: np.zeros((m, m), bool) for b in BLOCKS}
      for i in range(m):
        for j in range(i + 1, m):
          t = ptype(i, j)
          block_mask[t][i, j] = block_mask[t][j, i] = True

      def value(subset):
        Sig = diag.copy()
        for b in subset:
          Sig = Sig + np.where(block_mask[b], off, 0.0)
        Sig = psd(Sig + 1e-6 * np.eye(m))
        Si = np.linalg.inv(Sig)
        vals = [float(np.sqrt(max((mus[a] - mus[b]) @ Si @ (mus[a] - mus[b]), 0.0)))
                for a, b in cond_pairs]
        return float(np.percentile(vals, 10))

      cache = {}
      def v(fs):
        if fs not in cache:
          cache[fs] = value(fs)
        return cache[fs]

      phi = {}
      for b in BLOCKS:
        others = [x for x in BLOCKS if x != b]
        tot = 0.0
        for k in range(len(others) + 1):
          for comb in itertools.combinations(others, k):
            tot += weights[len(comb)] * (v(frozenset(comb) | {b}) - v(frozenset(comb)))
        phi[b] = tot

      # sign rule alignment per block, |r| weighted over member and condition pairs
      sdv = np.sqrt(np.diag(S))
      align = {b: [0.0, 0.0] for b in BLOCKS}       # [helpful weight, total weight]
      for i in range(m):
        for j in range(i + 1, m):
          if sdv[i] < 1e-9 or sdv[j] < 1e-9:
            continue
          rij = S[i, j] / (sdv[i] * sdv[j])
          t = ptype(i, j)
          for a, b2 in cond_pairs:
            sig = (mus[a, i] - mus[b2, i]) * (mus[a, j] - mus[b2, j])
            w = abs(rij)
            align[t][1] += w
            if rij * sig < 0:
              align[t][0] += w

      out = dict(original_network_idx=int(net), panel=r['panel'], accuracy=r['accuracy'],
                 n_P=cls.count('P'), n_D=cls.count('D'), n_U=cls.count('U'),
                 v_diag=v(frozenset()), v_full=v(frozenset(BLOCKS)))
      for b in BLOCKS:
        out[f'phi_{b}'] = phi[b]
        out[f'help_{b}'] = align[b][0] / align[b][1] if align[b][1] > 0 else np.nan
        out[f'w_{b}'] = align[b][1]
      rows.append(out)
      print(f"net {net} {r['panel']:16s} gain {out['v_full']-out['v_diag']:+6.2f}  "
            f"PP {phi['PP']:+5.2f} PD {phi['PD']:+5.2f} DD {phi['DD']:+5.2f} UX {phi['UX']:+5.2f}",
            flush=True)

  outp = pathlib.Path(args.output_csv)
  outp.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(outp, index=False)
  print(f'wrote {outp}')


if __name__ == '__main__':
  main()
