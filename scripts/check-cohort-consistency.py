#!/usr/bin/env python3
'''Are the cohorts of the rho sweep the same network ensemble?

The connectivity arrays were computed once from the batch 1 experiment. Reusing
them for another cohort would silently pair panels with the wrong network, so
this compares node degrees of network 0 across cohorts, and checks the stored
arrays against the networks CSV they claim to describe.

Findings: rho1.0 and rho0.5 reproduce the stored arrays exactly, 41641 edges,
while rho0.75-b4 and rho0.75-b5 are different ensembles, 57821 and 63365 edges.
So the eps 0.5 column of the connectivity figure needs its own arrays, which
compute-connectivity-arrays.py can rebuild from a seeded 1 step rerun of
perform_experiment, whose drug targets reproduce the stored ones exactly.

Run on the cluster from the repo root:
  python scripts/check-cohort-consistency.py
'''
import glob
import numpy as np, pandas as pd
conn = np.load('data/sensitivity/connectivity-arrays.npz')
ci, co = conn['in_deg'][0], conn['out_deg'][0]
NS = '/n/netscratch/nowak/Lab/dbrewster/boolean/drug-rho-sweep'
files = {'rho1.0': glob.glob(f'{NS}/rho1.0/derived/networks-*.csv'),
         'rho0.75-b4': glob.glob(f'{NS}/rho0.75-b4/derived/networks-*.csv'),
         'rho0.75-b5': glob.glob(f'{NS}/rho0.75-b5/derived/networks-*.csv'),
         'rho0.5': glob.glob(f'{NS}/rho0.5/derived/networks-*.csv')}
def deg0(path):
  parts = []
  for ch in pd.read_csv(path, chunksize=400_000):
    parts.append(ch[ch.original_network_idx == 0])
    if (ch.original_network_idx > 0).any():
      break
  e = pd.concat(parts)
  return (np.bincount(e.target.to_numpy(), minlength=5000),
          np.bincount(e.source.to_numpy(), minlength=5000))
print('conn arrays: nets', len(conn['networks']), 'in0 sum', ci.sum(), 'out0 sum', co.sum(), flush=True)
degs = {}
for tag, paths in files.items():
  if not paths:
    print(tag, 'NO networks csv', flush=True)
    continue
  try:
    inn, out = deg0(paths[0])
    degs[tag] = (inn, out)
    print(f'{tag}: file {paths[0].split("/")[-1]} in==conn {bool((inn==ci).all())} '
          f'out==conn {bool((out==co).all())} edges {out.sum()}', flush=True)
  except Exception as ex:
    print(tag, 'ERROR', type(ex).__name__, ex, flush=True)
tags = list(degs)
for i in range(len(tags)):
  for j in range(i+1, len(tags)):
    a, b = degs[tags[i]], degs[tags[j]]
    print(f'{tags[i]} vs {tags[j]}: in_equal {bool((a[0]==b[0]).all())} out_equal {bool((a[1]==b[1]).all())}', flush=True)
print('PHASE-A-DONE', flush=True)
