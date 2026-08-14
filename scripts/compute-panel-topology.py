'''Panel-level topology of evolved k=8 reporter subsets vs random panels.

For each network and each GA panel (per rho), and for size-matched random
panels, computes:
  - pairwise distances between panel members on the undirected graph
  - upstream coverage: fraction of nodes within r directed hops of the
    nearest panel member, following edge direction toward the panel
    (the set of nodes whose activity can reach a reporter in r steps)
  - downstream coverage: same but following edges away from the panel

Writes a tidy CSV with one row per (rho, network, panel).

Run on the cluster from the repo root:
  python scripts/compute-panel-topology.py \
    --networks-file data/drug-fixed-targets-v5/N5000/derived/networks-1771990942417.csv \
    --ga rho0.99=data/drug-fixed-targets-v5/N5000/ga-results-v5/combined-full.csv \
         rho0.5=data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
         rho0.9=data/drug-rho-sweep/rho0.9/ga-results/combined-full.csv \
    --num-random 50 \
    --out data/sensitivity/panel-topology.csv
'''
import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

K = 8
R_MAX = 10


def load_ga_subsets(ga_csv, k=K):
  ga = pd.read_csv(ga_csv)
  ga = ga[ga['max_num_features'] == k]
  final_idx = ga.groupby('original_network_idx')['generation'].idxmax()
  out = {}
  for _, r in ga.loc[final_idx].iterrows():
    nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
    if len(set(nodes)) == k:
      out[int(r['original_network_idx'])] = sorted(nodes)
  return out


def panel_metrics(panel, adj, adj_rev, adj_und, n):
  panel = np.asarray(panel)
  d_und = csgraph.dijkstra(adj_und, unweighted=True, indices=panel, limit=np.inf)
  sub = d_und[:, panel]
  iu = np.triu_indices(len(panel), 1)
  pairs = sub[iu]
  finite = pairs[np.isfinite(pairs)]

  row = {
    'mean_pair_dist': float(finite.mean()) if len(finite) else np.nan,
    'median_pair_dist': float(np.median(finite)) if len(finite) else np.nan,
    'disconnected_pairs': int(np.isinf(pairs).sum()),
  }
  for name, a in [('up', adj_rev), ('down', adj)]:
    d = csgraph.dijkstra(a, unweighted=True, indices=panel, limit=R_MAX + 1)
    nearest = d.min(axis=0)
    for r in range(1, R_MAX + 1):
      row[f'{name}_cov_r{r}'] = float((nearest <= r).mean())
  return row


def main(args):
  df = pd.read_csv(args.networks_file)
  nets = sorted(df['original_network_idx'].unique())
  n = args.network_size
  rng = np.random.default_rng(args.seed)

  ga_panels = {}
  for spec in args.ga:
    label, path = spec.split('=', 1)
    ga_panels[label] = load_ga_subsets(path)

  rows = []
  for net in nets:
    e = df[df['original_network_idx'] == net]
    src = e['source'].to_numpy(np.int64)
    tgt = e['target'].to_numpy(np.int64)
    ones = np.ones(len(src), dtype=np.int8)
    adj = sparse.csr_matrix((ones, (src, tgt)), shape=(n, n))
    adj_rev = adj.T.tocsr()
    adj_und = ((adj + adj_rev) > 0).astype(np.int8).tocsr()

    for label, panels in ga_panels.items():
      if net in panels:
        row = panel_metrics(panels[net], adj, adj_rev, adj_und, n)
        row.update({'rho': label, 'network': net, 'panel': 'genetic'})
        rows.append(row)

    for j in range(args.num_random):
      panel = rng.choice(n, size=K, replace=False)
      row = panel_metrics(panel, adj, adj_rev, adj_und, n)
      row.update({'rho': 'random', 'network': net, 'panel': f'random-{j}'})
      rows.append(row)
    print(f'net {net} done', flush=True)

  out = pd.DataFrame(rows)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  out.to_csv(args.out, index=False)
  print(f'saved {args.out} shape={out.shape}')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--networks-file', type=str, required=True)
  p.add_argument('--ga', type=str, nargs='+', required=True,
                 help='label=path pairs of GA combined-full.csv files')
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--num-random', type=int, default=50)
  p.add_argument('--seed', type=int, default=2025)
  p.add_argument('--out', type=str, required=True)
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
