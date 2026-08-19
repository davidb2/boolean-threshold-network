#!/usr/bin/env python3
'''Does member redundancy explain why evolved panels beat all dormant ones?

The naive Bayes combination of single member d primes assumes members err
independently. Under that assumption the margin designed dormant panels
dominate the evolved panels on every statistic of the pairwise d prime
distribution and still classify worse, so either the assumption fails or
the d prime picture is wrong. This script drops the assumption.

For each panel and each pair of alternatives it computes the Mahalanobis
discriminability of the two conditions from single trials,

  dM(q, q') = sqrt( (mu_q - mu_q')' Sigma^{-1} (mu_q - mu_q') ),

where mu are the condition means of the m member readings, one reading per
replicate, and Sigma is the pooled within condition covariance of the
members, shrunk halfway to its diagonal for stability with ten replicates.
It also reports the mean absolute off diagonal correlation between members,
the direct measure of redundancy.

If redundancy is the answer, the dormant panels should show larger member
correlations, their Mahalanobis discriminability should fall below the
naive Bayes value by more than the evolved panels' does, and the corrected
statistic should order the panel designs by accuracy.

Usage:
  python scripts/panel-redundancy-dprime.py \
    --states-file data/drug-fixed-targets-v7/N5000/derived/states-*.csv \
    --panels-csv data/panel-design/rho0.5/combined.csv \
    --output-csv data/panel-design/rho0.5/redundancy.csv
'''
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--panels-csv', type=str, required=True)
  p.add_argument('--output-csv', type=str, required=True)
  args = p.parse_args()

  panels = pd.read_csv(args.panels_csv)
  df = pd.read_csv(args.states_file, index_col=0).reset_index()
  df = df.rename(columns={'drug_name': 'Drug'})

  rows = []
  for net, g in panels.groupby('original_network_idx'):
    sub = df[df.original_network_idx == net]
    drugs = sorted(sub['Drug'].unique())
    for _, r in g.iterrows():
      members = [f'node-{x}' for x in r['nodes'].split()]
      m = len(members)
      # one reading per replicate: the member states averaged over snapshots
      reads = {d: sub[sub['Drug'] == d].groupby('initial_condition_idx')[members]
                   .mean().to_numpy() for d in drugs}
      mus = {d: v.mean(axis=0) for d, v in reads.items()}
      # pooled within condition covariance, shrunk halfway to its diagonal
      centered = np.vstack([v - v.mean(axis=0) for v in reads.values()])
      S = centered.T @ centered / max(len(centered) - len(drugs), 1)
      Sigma = 0.5 * S + 0.5 * np.diag(np.diag(S))
      Sigma += 1e-6 * np.eye(m)
      Sinv = np.linalg.inv(Sigma)
      # member redundancy: mean absolute off diagonal correlation
      sd = np.sqrt(np.diag(S))
      with np.errstate(divide='ignore', invalid='ignore'):
        corr = S / np.outer(sd, sd)
      off = corr[~np.eye(m, dtype=bool)]
      mean_abs_r = float(np.nanmean(np.abs(off)))
      # Mahalanobis and naive Bayes discriminability for every pair
      dM, dNB = [], []
      for a, b in itertools.combinations(drugs, 2):
        diff = mus[a] - mus[b]
        dM.append(float(np.sqrt(diff @ Sinv @ diff)))
        with np.errstate(divide='ignore', invalid='ignore'):
          z = np.where(sd > 1e-9, diff / sd, 0.0)
        dNB.append(float(np.sqrt((z ** 2).sum())))
      dM, dNB = np.array(dM), np.array(dNB)
      rows.append(dict(original_network_idx=net, panel=r['panel'],
                       accuracy=r['accuracy'], mean_abs_r=mean_abs_r,
                       min_dM=dM.min(), p10_dM=np.percentile(dM, 10),
                       median_dM=np.median(dM),
                       min_dNB=dNB.min(), median_dNB=np.median(dNB)))
      print(f'net {net} {r["panel"]:16s} |r|={mean_abs_r:.3f} '
            f'min dM {dM.min():6.2f}  median dM {np.median(dM):6.2f}', flush=True)

  out = pathlib.Path(args.output_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(out, index=False)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
