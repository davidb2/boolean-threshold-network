'''Per-drug per-node sensitivity S[network, drug, node] from a states CSV.

S is the mean |state - control| over initial conditions and stored steps,
resolved by drug instead of averaged as in compute-b-array.py.

Usage:
  python scripts/compute-per-drug-sensitivity.py \
    --states-file data/.../states-XXX.csv \
    --network-size 5000 \
    --out data/sensitivity/S-perdrug-rho0.99.npz
'''
import argparse
import os

import numpy as np
import pandas as pd


def main(args):
  node_cols = [f'node-{i}' for i in range(args.network_size)]
  df = pd.read_csv(args.states_file)
  networks = sorted(df['original_network_idx'].unique())
  drugs = sorted(
    [d for d in df['drug_name'].unique() if d != 'control'],
    key=lambda x: int(x.split('-')[1]),
  )
  S = np.zeros((len(networks), len(drugs), args.network_size), dtype=np.float32)
  for row, net in enumerate(networks):
    dnet = df[df['original_network_idx'] == net]
    ctrl = (
      dnet[dnet['drug_name'] == 'control']
      .set_index(['initial_condition_idx', 'step_num'])[node_cols]
      .astype(np.float64).sort_index()
    )
    for di, drug in enumerate(drugs):
      treated = (
        dnet[dnet['drug_name'] == drug]
        .set_index(['initial_condition_idx', 'step_num'])[node_cols]
        .astype(np.float64).sort_index()
      )
      S[row, di] = np.abs(treated.to_numpy() - ctrl.to_numpy()).mean(axis=0)
    print(f'net {net} done', flush=True)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  np.savez_compressed(args.out, S=S, networks=np.array(networks),
                      drugs=np.array(drugs))
  print(f'saved {args.out} shape={S.shape}')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--out', type=str, required=True)
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
