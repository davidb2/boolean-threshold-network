'''Control-run activity and within-panel information redundancy.

Two analyses on one pass over a states CSV:

1. Per-node activity in the CONTROL trajectories alone: the variance of
   the binary state across initial conditions and stored late steps.
   Frozen nodes cannot respond to shocks, so activity should predict
   sensitivity B without ever looking at the shocked runs.

2. Within-panel redundancy: pairwise mutual information and correlation
   between panel members' states over all late-time snapshots (all
   perturbation classes pooled), for the evolved panel, random panels,
   and sensitivity-matched random panels. Tests whether evolved panels
   minimize duplicate information.

Run on the cluster from the repo root, once per rho with matching files:
  python scripts/compute-info-redundancy.py \
    --states-file data/drug-fixed-targets-v5/N5000/derived/states-1771990942417.csv \
    --ga-file data/drug-fixed-targets-v5/N5000/ga-results-v5/combined-full.csv \
    --b-file data/sensitivity/B-rho0.99.npz \
    --rho 0.99 \
    --activity-out data/sensitivity/activity-rho0.99.npz \
    --redundancy-out data/sensitivity/redundancy-rho0.99.csv
'''
import argparse
import os

import numpy as np
import pandas as pd

K = 8
N_RAND = 50


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def load_ga_subsets(ga_csv):
  ga = pd.read_csv(ga_csv)
  ga = ga[ga['max_num_features'] == K]
  final_idx = ga.groupby('original_network_idx')['generation'].idxmax()
  out = {}
  for _, r in ga.loc[final_idx].iterrows():
    nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
    if len(set(nodes)) == K:
      out[int(r['original_network_idx'])] = sorted(nodes)
  return out


def pair_mi(x, y):
  '''Plug-in mutual information of two binary series, in bits.'''
  n = len(x)
  mi = 0.0
  for a in (0, 1):
    px = (x == a).mean()
    if px == 0:
      continue
    for b_ in (0, 1):
      py = (y == b_).mean()
      pxy = ((x == a) & (y == b_)).mean()
      if pxy > 0 and py > 0:
        mi += pxy * np.log2(pxy / (px * py))
  return mi


def panel_redundancy(states, nodes):
  X = states[:, nodes]
  mis, cors = [], []
  for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
      mis.append(pair_mi(X[:, i], X[:, j]))
      sx, sy = X[:, i].std(), X[:, j].std()
      cors.append(abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) if sx > 0 and sy > 0 else 0.0)
  p1 = X.mean(axis=0)
  ok = (p1 > 0) & (p1 < 1)
  marg = np.zeros(len(nodes))
  marg[ok] = -(p1[ok] * np.log2(p1[ok]) + (1 - p1[ok]) * np.log2(1 - p1[ok]))
  return dict(mean_mi=float(np.mean(mis)), max_mi=float(np.max(mis)),
              mean_abscorr=float(np.mean(cors)), mean_marg_H=float(marg.mean()))


def main(args):
  df = pd.read_csv(args.states_file)
  node_cols = [f'node-{i}' for i in range(args.network_size)]
  b_data = np.load(args.b_file)
  B, bnets = b_data['B'], [int(x) for x in b_data['networks']]
  cut = antimode(B)
  panels = load_ga_subsets(args.ga_file)
  rng = np.random.default_rng(args.seed)

  nets = sorted(df['original_network_idx'].unique())
  activity = np.zeros((len(nets), args.network_size))
  rows = []
  for i, net in enumerate(nets):
    dnet = df[df['original_network_idx'] == net]
    ctrl = dnet[dnet['drug_name'] == 'control'][node_cols].to_numpy(dtype=np.float64)
    activity[i] = ctrl.var(axis=0)
    if net not in panels:
      continue
    states = dnet[node_cols].to_numpy(dtype=np.int8)
    bi = bnets.index(net)
    brow = B[bi]
    nodes = panels[net]
    r = panel_redundancy(states, nodes)
    r.update(rho=args.rho, network=net, panel='evolved')
    rows.append(r)
    sens_pool = np.flatnonzero(brow > cut)
    insens_pool = np.flatnonzero(brow <= cut)
    n_s = int((brow[nodes] > cut).sum())
    for j in range(N_RAND):
      rn = rng.choice(args.network_size, K, replace=False)
      r = panel_redundancy(states, list(rn))
      r.update(rho=args.rho, network=net, panel=f'random-{j}')
      rows.append(r)
      mn = np.concatenate([rng.choice(sens_pool, n_s, replace=False),
                           rng.choice(insens_pool, K - n_s, replace=False)])
      r = panel_redundancy(states, list(mn))
      r.update(rho=args.rho, network=net, panel=f'matched-{j}')
      rows.append(r)
    print(f'net {net} done', flush=True)

  os.makedirs(os.path.dirname(args.activity_out) or '.', exist_ok=True)
  np.savez_compressed(args.activity_out, activity=activity, networks=np.array(nets))
  pd.DataFrame(rows).to_csv(args.redundancy_out, index=False)
  print(f'saved {args.activity_out} and {args.redundancy_out}')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--rho', type=float, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--seed', type=int, default=31)
  p.add_argument('--activity-out', type=str, required=True)
  p.add_argument('--redundancy-out', type=str, required=True)
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
