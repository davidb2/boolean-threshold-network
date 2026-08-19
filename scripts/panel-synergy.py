#!/usr/bin/env python3
'''Is there an information theoretic conspiracy the covariance test missed?

The covariance attribution showed member correlations contribute about one
percent of the predictive statistic, but covariance sees only linear
dependence and the node states are Boolean, where synergy can live entirely
in higher order structure, the XOR being the canonical case. This script
measures the entropic analogue directly.

For every member pair of every panel, with readings binarized per replicate:

  interaction information  II = I(Q; Xi, Xj) - I(Q; Xi) - I(Q; Xj)
      positive means the pair carries MORE shock information together than
      its members do separately, which is synergy, the information
      theoretic definition of a conspiracy
  noise dependence         I(Xi; Xj | Q), the entropic noise correlation
  signal dependence        I(Xi; Xj), redundancy of what the members say

Plug in mutual information estimates on 110 trials are biased upward, so
every estimate has the mean of a label permutation null subtracted, which
removes the bias exactly to first order. Results are decomposed by member
pair type, promiscuous with promiscuous, cross class, dormant with dormant,
so the question "do the classes conspire" gets a direct answer.

Usage:
  python scripts/panel-synergy.py \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --panels-csv data/panel-design/rho0.5/combined.csv \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --b-file data/sensitivity/B-rho0.5.npz \
    --output-csv data/panel-design/rho0.5/synergy.csv
'''
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

SPLIT = 6
NPERM = 200


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def mi(x, y, nx, ny):
  '''Plug in mutual information in bits for small discrete variables.'''
  joint = np.zeros((nx, ny))
  np.add.at(joint, (x, y), 1.0)
  joint /= joint.sum()
  px = joint.sum(axis=1, keepdims=True)
  py = joint.sum(axis=0, keepdims=True)
  with np.errstate(divide='ignore', invalid='ignore'):
    t = joint * np.log2(joint / (px @ py))
  return float(np.nansum(t))


def corrected(x, y, nx, ny, rng):
  '''Plug in MI minus the mean of a label permutation null.'''
  raw = mi(x, y, nx, ny)
  null = np.mean([mi(x, rng.permutation(y), nx, ny) for _ in range(NPERM)])
  return raw - null


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
  cut = antimode(np.load(args.b_file)['B'])

  panels = pd.read_csv(args.panels_csv)
  df = pd.read_csv(args.states_file, index_col=0).reset_index()
  df = df.rename(columns={'drug_name': 'Drug'})
  rng = np.random.default_rng(0)

  rows = []
  for net, g in panels.groupby('original_network_idx'):
    si = snets.index(int(net))
    n_row = (S_all[si] >= cut).sum(axis=1)
    sub = df[df.original_network_idx == net]
    drugs = sorted(sub['Drug'].unique())
    for _, r in g.iterrows():
      nodes = [int(x) for x in r['nodes'].split()]
      members = [f'node-{x}' for x in nodes]
      cls = ['U' if n_row[x] == 0 else 'P' if n_row[x] >= SPLIT else 'D' for x in nodes]
      # one binary reading per replicate and condition
      X, Q = [], []
      for qi, d in enumerate(drugs):
        rep = sub[sub['Drug'] == d].groupby('initial_condition_idx')[members].mean()
        X.append((rep.to_numpy() > 0.5).astype(int))
        Q.extend([qi] * len(rep))
      X = np.vstack(X)
      Q = np.array(Q)
      nq = len(drugs)

      Iq = [corrected(X[:, i], Q, 2, nq, rng) for i in range(len(nodes))]
      agg = {}
      for i, j in itertools.combinations(range(len(nodes)), 2):
        a, b = sorted((cls[i], cls[j]))
        t = 'UX' if 'U' in (a, b) else {('P','P'):'PP',('D','P'):'PD',('D','D'):'DD'}[(a,b)]
        pair = X[:, i] * 2 + X[:, j]
        Iq_pair = corrected(pair, Q, 4, nq, rng)
        II = Iq_pair - Iq[i] - Iq[j]
        Isig = corrected(X[:, i], X[:, j], 2, 2, rng)
        # noise MI: average the within condition dependence
        noise = []
        for qi in range(nq):
          m = Q == qi
          if m.sum() >= 4:
            noise.append(corrected(X[m, i], X[m, j], 2, 2, rng))
        d0 = agg.setdefault(t, dict(II=[], noise=[], sig=[]))
        d0['II'].append(II)
        d0['noise'].append(np.mean(noise) if noise else np.nan)
        d0['sig'].append(Isig)

      out = dict(original_network_idx=int(net), panel=r['panel'],
                 accuracy=r['accuracy'], sum_Iq=float(np.sum(Iq)),
                 n_P=cls.count('P'), n_D=cls.count('D'), n_U=cls.count('U'))
      for t in ['PP', 'PD', 'DD', 'UX']:
        d0 = agg.get(t, None)
        out[f'II_{t}'] = float(np.mean(d0['II'])) if d0 else np.nan
        out[f'noise_{t}'] = float(np.nanmean(d0['noise'])) if d0 else np.nan
        out[f'sig_{t}'] = float(np.mean(d0['sig'])) if d0 else np.nan
        out[f'npairs_{t}'] = len(d0['II']) if d0 else 0
      rows.append(out)
      print(f"net {net} {r['panel']:16s} II  PP {out['II_PP']:+.3f}  PD {out['II_PD']:+.3f}  "
            f"DD {out['II_DD']:+.3f}", flush=True)

  outp = pathlib.Path(args.output_csv)
  outp.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(outp, index=False)
  print(f'wrote {outp}')


if __name__ == '__main__':
  main()
