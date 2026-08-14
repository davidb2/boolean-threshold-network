'''Per-node connectivity arrays for the reporter-subset connectivity figure.

From the networks CSV and the raw protobuf experiment file, computes for
every network: in-degree, out-degree, the shock target mask per drug, and
the directed graph distance from the nearest target of each drug to every
node (multi-source BFS along edge direction). Saves one compressed npz.

Run on the cluster from the repo root:
  python scripts/compute-connectivity-arrays.py \
    --networks-file data/drug-fixed-targets-v5/N5000/derived/networks-1771990942417.csv \
    --pb-dir data/drug-fixed-targets-v5/N5000/raw \
    --network-size 5000 \
    --out data/sensitivity/connectivity-arrays.npz
'''
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.abspath('python_generated'))


def load_targets(pb_dir, num_networks, num_drugs, n):
  import message_pb2
  mask = np.zeros((num_networks, num_drugs, n), dtype=bool)
  for path in sorted(glob.glob(f'{pb_dir}/*.pb')):
    exp = message_pb2.Experiment()
    with open(path, 'rb') as f:
      exp.ParseFromString(f.read())
    for res in exp.results:
      net = res.network_idx
      drug_idx = 0
      for pert in res.perturbations:
        if pert.name == 'control':
          continue
        sources = {ep.source for ep in pert.edge_perturbations}
        mask[net, drug_idx, list(sources)] = True
        drug_idx += 1
  return mask


def multi_source_bfs(adj_csr, sources, n):
  dist = np.full(n, -1, dtype=np.int16)
  frontier = np.array(sorted(sources), dtype=np.int64)
  dist[frontier] = 0
  level = 0
  while len(frontier):
    level += 1
    nxt = np.unique(np.concatenate([
      adj_csr.indices[adj_csr.indptr[u]:adj_csr.indptr[u + 1]] for u in frontier
    ])) if len(frontier) else np.array([], dtype=np.int64)
    nxt = nxt[dist[nxt] == -1]
    dist[nxt] = level
    frontier = nxt
  return dist


def main(args):
  n = args.network_size
  df = pd.read_csv(args.networks_file)
  nets = sorted(df['original_network_idx'].unique())
  num_networks = len(nets)

  in_deg = np.zeros((num_networks, n), dtype=np.int32)
  out_deg = np.zeros((num_networks, n), dtype=np.int32)
  adjs = {}
  for i, net in enumerate(nets):
    e = df[df['original_network_idx'] == net]
    src = e['source'].to_numpy(np.int64)
    tgt = e['target'].to_numpy(np.int64)
    out_deg[i] = np.bincount(src, minlength=n)
    in_deg[i] = np.bincount(tgt, minlength=n)
    adjs[net] = sparse.csr_matrix(
      (np.ones(len(src), dtype=np.int8), (src, tgt)), shape=(n, n),
    )
    print(f'degrees {i + 1}/{num_networks}', flush=True)

  targets = load_targets(args.pb_dir, num_networks, args.num_drugs, n)
  print('targets parsed', flush=True)

  dist = np.full((num_networks, args.num_drugs, n), -1, dtype=np.int16)
  for i, net in enumerate(nets):
    for d in range(args.num_drugs):
      srcs = np.flatnonzero(targets[i, d])
      if len(srcs):
        dist[i, d] = multi_source_bfs(adjs[net], srcs, n)
    print(f'bfs {i + 1}/{num_networks}', flush=True)

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  np.savez_compressed(
    args.out,
    networks=np.array(nets), in_deg=in_deg, out_deg=out_deg,
    targets=targets, dist_from_targets=dist,
  )
  print(f'saved {args.out}')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--networks-file', type=str, required=True)
  p.add_argument('--pb-dir', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--num-drugs', type=int, default=10)
  p.add_argument('--out', type=str, required=True)
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
