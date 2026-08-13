'''Compute the per-node sensitivity array B[network, node] from a states CSV.

B is the mean |state - control| over drugs, initial conditions, and stored
final steps. Saves an npz with keys B and networks, matching the cache format
of sensitive-nodes-vs-rho.ipynb and the input format of node-ablation-k8.py.

Usage:
  python scripts/compute-b-array.py \
    --states-file data/.../derived/states-XXX.csv \
    --network-size 5000 \
    --out data/sensitivity/B-rho0.9.npz
'''
import argparse
import os

import numpy as np
import pandas as pd


def compute_B(states_csv, n):
  node_cols = [f'node-{i}' for i in range(n)]
  df = pd.read_csv(states_csv)
  networks = sorted(df['original_network_idx'].unique())
  drugs = sorted(
    [d for d in df['drug_name'].unique() if d != 'control'],
    key=lambda x: int(x.split('-')[1]),
  )
  B = np.zeros((len(networks), n), dtype=np.float64)
  for row, net in enumerate(networks):
    df_net = df[df['original_network_idx'] == net]
    ctrl = (
      df_net[df_net['drug_name'] == 'control']
      .set_index(['initial_condition_idx', 'step_num'])[node_cols]
      .astype(np.float64)
      .sort_index()
    )
    acc = np.zeros(n, dtype=np.float64)
    for drug in drugs:
      treated = (
        df_net[df_net['drug_name'] == drug]
        .set_index(['initial_condition_idx', 'step_num'])[node_cols]
        .astype(np.float64)
        .sort_index()
      )
      assert treated.shape == ctrl.shape, f'shape mismatch net={net} drug={drug}'
      acc += np.abs(treated.to_numpy() - ctrl.to_numpy()).mean(axis=0)
    B[row] = acc / len(drugs)
    print(f'  [{row + 1}/{len(networks)}] net {net} done', flush=True)
  return B, np.array(networks)


def main(args):
  if os.path.exists(args.out) and not args.force:
    print(f'{args.out} already exists, skipping (use --force to recompute)')
    return
  print(f'computing B from {args.states_file} ...', flush=True)
  B, nets = compute_B(args.states_file, args.network_size)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  np.savez_compressed(args.out, B=B, networks=nets)
  print(f'saved {args.out}  B shape={B.shape}  mean={B.mean():.4f}')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--out', type=str, required=True)
  p.add_argument('--force', action='store_true')
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
