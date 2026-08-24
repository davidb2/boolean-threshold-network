#!/usr/bin/env python3
'''Which inputs carry the whole panel size ladder?

A selection strategy whose results cover a single panel size draws a one point
line, which matplotlib renders as nothing at all. That is how the genetic
algorithm curve and the eps 1 random curve were silently dropped from Figure 7
in one re-render. plot-selection-strategies-figure.py now refuses such an input
through check_ladder, and this script reports which sources are safe to pass.

Findings:

  genetic algorithm  the raw clean search outputs cover only m = 8 at eps 0.5
                     and 1, so the curve needs rescored/ga-clean-rescored-all-*
  random             only rho1.0 and rho0.75-b4 carry m = 1 to 128 in the rho
                     sweep. At rho 0.5 the sweep ran random selection at m = 8
                     only, so the eps 1 baseline comes from the original v7
                     sweep, whose sizes past the x limit of 150 are clipped

Run on the cluster from the repo root:
  python scripts/check-figure-inputs.py
'''
import pathlib

import numpy as np
import pandas as pd

NS = '/n/netscratch/nowak/Lab/dbrewster/boolean'
LEVELS = [('rho1.0', 0), ('rho0.75-b4', 0.5), ('rho0.5', 1)]


def ladder_of_csv(path):
  d = pd.read_csv(path)
  acol = 'accuracy' if 'accuracy' in d.columns else 'best_accuracy'
  if 'max_num_features' not in d.columns:
    return 'no size column', None
  sizes = sorted(int(x) for x in d['max_num_features'].unique())
  sub = d[d.max_num_features == 8]
  if 'generation' in sub.columns and len(sub):
    sub = sub.loc[sub.groupby('original_network_idx')['generation'].idxmax()]
  m8 = sub[acol].mean() if len(sub) else float('nan')
  return sizes, m8


def ladder_of_dir(d):
  p = pathlib.Path(d)
  if not p.exists():
    return 'MISSING', None
  fs = [f for f in sorted(p.glob('*-full.csv')) if f.name != 'combined-full.csv']
  if not fs:
    return 'no per network *-full.csv', None
  df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
  per = df.groupby('max_num_features')['original_network_idx'].nunique()
  return list(int(x) for x in per.index), per.to_dict()


def main():
  print('=== genetic algorithm sources')
  for tag, eps in LEVELS:
    for path in [f'data/drug-rho-sweep/{tag}/ga-results-clean/combined-full.csv',
                 f'{NS}/rescored/ga-clean-rescored-all-{tag}.csv']:
      if not pathlib.Path(path).exists():
        print(f'   eps {eps:<4g} MISSING {path}')
        continue
      sizes, m8 = ladder_of_csv(path)
      flag = 'ONE SIZE, curve cannot be drawn' if len(sizes) < 2 else 'full ladder'
      print(f'   eps {eps:<4g} {pathlib.Path(path).name:38s} {flag}, '
            f'sizes {sizes}, m8 {m8:.4f}')

  print()
  print('=== random baseline directories')
  roots = [pathlib.Path('data')]
  cands = sorted(p for root in roots if root.exists()
                 for p in root.rglob('*random*') if p.is_dir())
  for p in cands:
    sizes, per = ladder_of_dir(p)
    if isinstance(sizes, str):
      continue
    flag = 'ONE SIZE' if len(sizes) < 2 else f'{len(sizes)} sizes'
    print(f'   {str(p):60s} {flag}, up to m {max(sizes)}')

  print()
  print('=== the eps 1 random curve, published values read off the figure')
  print('   m1 0.109  m2 0.155  m4 0.181  m8 0.391  m16 0.554  m32 0.754')
  for d in ['data/drug-fixed-targets-v7/N5000/random-results-v7',
            'data/drug-rho-sweep/rho0.5/random-results',
            'data/drug-fixed-targets-v5/N5000/random-results-v5']:
    p = pathlib.Path(d)
    if not p.exists():
      continue
    fs = [f for f in sorted(p.glob('*-full.csv')) if f.name != 'combined-full.csv']
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    per = df.groupby(['max_num_features', 'original_network_idx'])['accuracy'].mean()
    c = per.groupby('max_num_features').mean()
    shown = {int(k): round(float(v), 3) for k, v in c.items() if k <= 32}
    print(f'   {p.name:26s} {shown}')


if __name__ == '__main__':
  main()
