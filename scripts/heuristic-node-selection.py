#!/usr/bin/env python3
'''Per-network heuristic node selection baselines.

Implements five selection strategies to compare against the genetic algorithm
and random selection, all scored with the same random forest evaluator
(classifier.train_and_test):

  sensitivity  top nodes by per-node Hamming distance from control (needs --b-file)
  in-degree    most regulated nodes (needs --networks-file)
  out-degree   biggest hubs (needs --networks-file)
  mmse         greedy covariance column selection on control snapshots
  jaccard      greedy coverage of distinct downstream influence sets (needs --networks-file)

Each strategy produces a ranking; prefixes of the ranking are evaluated at each
feature size. Output matches random-node-selection.py ({i}-full.csv + .done).

Usage:
  python scripts/heuristic-node-selection.py \
    --strategy sensitivity \
    --original-network-idx 0 \
    --states-file data/drug-fixed-targets-v5/N5000/derived/states-*.csv \
    --b-file data/sensitivity/B-rho0.99.npz \
    --network-size 5000 \
    --feature-sizes 1 2 4 8 16 32 64 128 \
    --output-dir data/selection-strategies/rho0.99/sensitivity-results
'''
import argparse
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from typing import *

from classifier import train_and_test


def load_edges(networks_file, network_idx):
  df = pd.read_csv(networks_file)
  df = df[df['original_network_idx'] == network_idx]
  return df[['source', 'target']].to_numpy(dtype=np.int64)


def rank_by_degree(edges, n, direction, rng):
  col = 0 if direction == 'out' else 1
  deg = np.bincount(edges[:, col], minlength=n)
  jitter = rng.random(n)
  return np.lexsort((jitter, -deg))


def rank_by_sensitivity(b_file, network_idx, n, rng):
  data = np.load(b_file)
  nets = [int(x) for x in data['networks']]
  b = data['B'][nets.index(network_idx)]
  assert len(b) == n
  jitter = rng.random(n)
  return np.lexsort((jitter, -b))


def rank_by_mmse(states_df, node_cols, k_max, ridge=1e-9, tol=1e-12):
  ctrl = states_df[states_df['Drug'] == 'control'][node_cols]
  X = ctrl.to_numpy(dtype=np.float64)
  X = X - X.mean(axis=0)
  sigma = (X.T @ X) / (X.shape[0] - 1)
  n = sigma.shape[0]
  sigma = sigma + ridge * np.eye(n)

  S = []
  selected = np.zeros(n, dtype=bool)
  diag = np.diag(sigma).copy()
  for _ in range(k_max):
    if not S:
      s_all = diag
      qnorm = (sigma ** 2).sum(axis=0)
    else:
      sigma_s = sigma[:, S]
      ainv = np.linalg.inv(sigma[np.ix_(S, S)])
      v = ainv @ sigma_s.T
      s_all = diag - np.einsum('ij,ji->i', sigma_s, v)
      resid = sigma - sigma_s @ v
      qnorm = (resid ** 2).sum(axis=0)
    gain = np.where((s_all > tol) & ~selected, qnorm / np.maximum(s_all, tol), -np.inf)
    j = int(np.argmax(gain))
    if not np.isfinite(gain[j]):
      break
    S.append(j)
    selected[j] = True
  return np.array(S, dtype=np.int64)


def rank_by_jaccard(edges, n, k_max, rng):
  succ = {}
  for s, t in edges:
    succ.setdefault(int(s), set()).add(int(t))
  sizes = {v: len(nb) for v, nb in succ.items()}
  covered = set()
  remaining = set(range(n))
  order = []
  jitter = rng.random(n)
  for _ in range(min(k_max, n)):
    best, best_key = None, None
    ca = len(covered)
    for v in remaining:
      nb = succ.get(v)
      if nb is None:
        dis = 0.0 if ca == 0 else 1.0
      else:
        inter = len(covered & nb)
        union = ca + sizes[v] - inter
        dis = 1.0 - (inter / union) if union else 0.0
      key = (dis, jitter[v])
      if best_key is None or key > best_key:
        best, best_key = v, key
    order.append(best)
    remaining.discard(best)
    if best in succ:
      covered |= succ[best]
  return np.array(order, dtype=np.int64)


def _score_one(task):
  features, network_idx, df = task
  perf, _ = train_and_test(
    df, num_trials=1, original_network_idx=network_idx, dep_vars=list(features),
  )
  return float(perf['correct'].mean())



def rank_by_influence(edges, n, k_max, rng):
  """Greedy influence maximization: cover the largest 2 hop downstream set."""
  out_nbrs = [[] for _ in range(n)]
  for s, t in edges:
    out_nbrs[int(s)].append(int(t))
  reach = []
  for v in range(n):
    r = np.zeros(n, dtype=bool)
    one = out_nbrs[v]
    r[one] = True
    for u in one:
      r[out_nbrs[u]] = True
    reach.append(r)
  covered = np.zeros(n, dtype=bool)
  jitter = rng.random(n)
  order, taken = [], np.zeros(n, dtype=bool)
  for _ in range(min(k_max, n)):
    gains = np.array([(-1.0 if taken[v] else float((reach[v] & ~covered).sum())) for v in range(n)])
    best = int(np.lexsort((jitter, -gains))[0])
    order.append(best)
    taken[best] = True
    covered |= reach[best]
  return np.array(order, dtype=np.int64)


def rank_by_upstream(edges, n, k_max, rng):
  """Greedy coverage of distinct upstream regulators (the colab smart variant)."""
  in_nbrs = [set() for _ in range(n)]
  for s, t in edges:
    in_nbrs[int(t)].add(int(s))
  seen = set()
  jitter = rng.random(n)
  taken = np.zeros(n, dtype=bool)
  order = []
  for step in range(min(k_max, n)):
    if step == 0:
      gains = np.array([(-1.0 if taken[v] else float(len(in_nbrs[v]))) for v in range(n)])
    else:
      gains = np.array([(-1.0 if taken[v] else float(len(in_nbrs[v] - seen))) for v in range(n)])
    best = int(np.lexsort((jitter, -gains))[0])
    order.append(best)
    taken[best] = True
    seen |= in_nbrs[best]
  return np.array(order, dtype=np.int64)


def pairwise_mi_matrix(X):
  """Pairwise MI in bits for binary columns of X, as in the anchor notebook."""
  n, N = X.shape
  X = X.astype(np.float64)
  p1 = X.mean(axis=0)
  p0 = 1 - p1
  p11 = (X.T @ X) / n
  p10 = p1[:, None] - p11
  p01 = p1[None, :] - p11
  p00 = 1 - p1[:, None] - p1[None, :] + p11

  def h(p):
    return np.where(p > 1e-12, -p * np.log2(np.where(p > 1e-12, p, 1.0)), 0.0)

  H_joint = h(p11) + h(p10) + h(p01) + h(p00)
  H_marg = h(p1) + h(p0)
  MI = H_marg[:, None] + H_marg[None, :] - H_joint
  np.fill_diagonal(MI, 0.0)
  return np.clip(MI, 0, None), H_marg


def rank_by_entropy_diversity(states_df, node_cols, k_max, beta, rng):
  """Greedy: high control state entropy, penalized by MI with the selected set."""
  X = states_df[states_df['Drug'] == 'control'][node_cols].to_numpy(dtype=np.int8)
  MI, H = pairwise_mi_matrix(X)
  n = len(node_cols)
  jitter = 1e-9 * rng.random(n)
  taken = np.zeros(n, dtype=bool)
  order = []
  mi_sum = np.zeros(n)
  for step in range(min(k_max, n)):
    red = mi_sum / step if step else 0.0
    scores = H + jitter - beta * red
    scores[taken] = -np.inf
    best = int(np.argmax(scores))
    order.append(best)
    taken[best] = True
    mi_sum += MI[:, best]
  return np.array(order, dtype=np.int64)


def plugin_conditional_entropy(y, panel_states, n_classes):
  n = len(y)
  _, state_idx = np.unique(panel_states, return_inverse=True)
  h = 0.0
  for si in range(state_idx.max() + 1):
    mask = state_idx == si
    p_s = mask.sum() / n
    counts = np.bincount(y[mask], minlength=n_classes)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    h += p_s * float(-(p * np.log2(p)).sum())
  return h


def greedy_infomax(X, y, n_classes, k_max, start_panel=(), candidates=None):
  """Greedy minimization of the plug in H(class | panel), notebook style."""
  n_nodes = X.shape[1]
  in_panel = np.zeros(n_nodes, dtype=bool)
  panel_states = np.zeros(len(y), dtype=np.int64)
  for a in start_panel:
    in_panel[a] = True
    panel_states = (panel_states << 1) | X[:, a].astype(np.int64)
  if candidates is None:
    candidates = range(n_nodes)
  order = list(start_panel)
  for _ in range(k_max - len(order)):
    best_h, best_node = np.inf, -1
    for j in candidates:
      if in_panel[j]:
        continue
      new_states = (panel_states << 1) | X[:, j].astype(np.int64)
      h = plugin_conditional_entropy(y, new_states, n_classes)
      if h < best_h:
        best_h, best_node = h, j
    panel_states = (panel_states << 1) | X[:, best_node].astype(np.int64)
    in_panel[best_node] = True
    order.append(best_node)
  return order


def drug_matrix(states_df, node_cols):
  drug_rows = states_df[states_df['Drug'] != 'control']
  drugs = sorted(drug_rows['Drug'].unique())
  d2i = {d: i for i, d in enumerate(drugs)}
  X = drug_rows[node_cols].to_numpy(dtype=np.int8)
  y = np.array([d2i[d] for d in drug_rows['Drug']])
  return X, y, len(drugs)


def rank_by_infomax(states_df, node_cols, k_max, b):
  # candidates are scanned in descending sensitivity so entropy ties resolve
  # toward sensitive nodes, matching the notebook's reporter_candidates order
  X, y, n_classes = drug_matrix(states_df, node_cols)
  candidates = np.argsort(-b)
  return np.array(greedy_infomax(X, y, n_classes, k_max, candidates=candidates),
                  dtype=np.int64)


def anchor_reporter_panel(states_df, node_cols, b, k, anchor_fraction, beta, rng):
  """Two phase anchor+reporter selection from the polished notebook.

  Phase 1: l anchors scored by -sensitivity - beta * mean MI with the set.
  Phase 2: k - l reporters greedily minimize H(Drug | panel).

  Two deliberate deviations from the notebook: the MI matrix is computed per
  network rather than pooled across networks (pooled MI mostly measures cross
  network heterogeneity, not the within network redundancy the penalty
  targets), and initial condition 0 is kept, consistent with the classifier
  and B array used everywhere else in the pipeline.
  """
  X_ctrl = states_df[states_df['Drug'] == 'control'][node_cols].to_numpy(dtype=np.int8)
  MI, _ = pairwise_mi_matrix(X_ctrl)
  n = len(node_cols)
  l = int(round(anchor_fraction * k))
  jitter = 1e-9 * rng.random(n)
  taken = np.zeros(n, dtype=bool)
  anchors = []
  mi_sum = np.zeros(n)
  for step in range(l):
    red = mi_sum / step if step else 0.0
    scores = -b - beta * red + jitter
    scores[taken] = -np.inf
    best = int(np.argmax(scores))
    anchors.append(best)
    taken[best] = True
    mi_sum += MI[:, best]
  X, y, n_classes = drug_matrix(states_df, node_cols)
  return greedy_infomax(X, y, n_classes, k, start_panel=anchors,
                        candidates=np.argsort(-b))




def anchor_sensitivity_panel(states_df, node_cols, b, k, anchor_fraction, beta, rng):
  """The simple two phase rule: MI diverse anchors, then the most sensitive
  nodes. No entropy machinery, only per node statistics, so it runs at any
  panel size."""
  X_ctrl = states_df[states_df['Drug'] == 'control'][node_cols].to_numpy(dtype=np.int8)
  MI, _ = pairwise_mi_matrix(X_ctrl)
  n = len(node_cols)
  l = int(round(anchor_fraction * k))
  jitter = 1e-9 * rng.random(n)
  taken = np.zeros(n, dtype=bool)
  panel = []
  mi_sum = np.zeros(n)
  for step in range(l):
    red = mi_sum / step if step else 0.0
    scores = -b - beta * red + jitter
    scores[taken] = -np.inf
    best = int(np.argmax(scores))
    panel.append(best)
    taken[best] = True
    mi_sum += MI[:, best]
  order = np.lexsort((rng.random(n), -b))
  for j in order:
    if len(panel) >= k:
      break
    if not taken[j]:
      panel.append(int(j))
      taken[j] = True
  return panel

def load_b_row(b_file, network_idx, n):
  data = np.load(b_file)
  nets = [int(x) for x in data['networks']]
  b = data['B'][nets.index(network_idx)]
  assert len(b) == n
  return b

def get_ranking(args, states_df, node_cols, k_max, rng):
  if args.strategy == 'sensitivity':
    return rank_by_sensitivity(args.b_file, args.original_network_idx, args.network_size, rng)
  if args.strategy in ('in-degree', 'out-degree'):
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_degree(edges, args.network_size, args.strategy.split('-')[0], rng)
  if args.strategy == 'mmse':
    return rank_by_mmse(states_df, node_cols, k_max)
  if args.strategy == 'jaccard':
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_jaccard(edges, args.network_size, k_max, rng)
  if args.strategy == 'influence':
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_influence(edges, args.network_size, k_max, rng)
  if args.strategy == 'upstream':
    edges = load_edges(args.networks_file, args.original_network_idx)
    return rank_by_upstream(edges, args.network_size, k_max, rng)
  if args.strategy == 'entropy-diversity':
    return rank_by_entropy_diversity(states_df, node_cols, k_max, args.beta, rng)
  if args.strategy == 'infomax':
    b = load_b_row(args.b_file, args.original_network_idx, args.network_size)
    return rank_by_infomax(states_df, node_cols, k_max, b)
  raise ValueError(f'unknown strategy {args.strategy}')


def main(args):
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  output_path = out_dir / f'{args.original_network_idx}-full.csv'
  done_path = out_dir / f'{args.original_network_idx}-full.done'
  if done_path.exists():
    return

  states_df = pd.read_csv(args.states_file, index_col=0)
  states_df = states_df.reset_index().rename(columns={'drug_name': 'Drug'})
  states_df = states_df[states_df['original_network_idx'] == args.original_network_idx]
  states_df = states_df.drop(columns=['original_network_idx', 'initial_condition_idx'])
  node_cols = [f'node-{i}' for i in range(args.network_size)]

  # entropy based strategies rely on the plug in estimator, which is only
  # reliable while 2^k is small next to the number of snapshots
  ENTROPY_MAX_K = 16
  entropy_family = args.strategy in ('infomax', 'anchor-reporter')  # anchor-sensitivity has no entropy step
  sizes = [k for k in args.feature_sizes if not (entropy_family and k > ENTROPY_MAX_K)]
  for k in set(args.feature_sizes) - set(sizes):
    print(f'skipping k={k}: beyond the plug in entropy validity cap', flush=True)

  k_max = max(sizes)
  rows = []
  tasks = []
  for trial in range(args.num_trials):
    rng = np.random.default_rng(args.seed + trial)
    if args.strategy in ('anchor-reporter', 'anchor-sensitivity'):
      b = load_b_row(args.b_file, args.original_network_idx, args.network_size)
      builder = (anchor_reporter_panel if args.strategy == 'anchor-reporter'
                 else anchor_sensitivity_panel)
      for k in sizes:
        panel = builder(states_df, node_cols, b, k,
                        args.anchor_fraction, args.beta, rng)
        tasks.append((trial, k, [f'node-{i}' for i in panel]))
    else:
      ranking = get_ranking(args, states_df, node_cols, k_max, rng)
      for k in sizes:
        if k > len(ranking):
          print(f'skipping k={k}: ranking only has {len(ranking)} nodes', flush=True)
          continue
        features = [f'node-{i}' for i in ranking[:k]]
        tasks.append((trial, k, features))

  with multiprocessing.Pool(processes=args.num_workers) as pool:
    accs = pool.map(
      _score_one,
      [(features, args.original_network_idx, states_df) for _, _, features in tasks],
    )

  for (trial, k, features), acc in zip(tasks, accs):
    rows.append({
      'original_network_idx': args.original_network_idx,
      'strategy': args.strategy,
      'max_num_features': k,
      'trial': trial,
      'accuracy': acc,
      'features': sorted(features),
    })
    print(f'k={k}, trial={trial}, accuracy={acc:.4f}', flush=True)

  output_path.unlink(missing_ok=True)
  pd.DataFrame(rows).to_csv(output_path, index=False)
  done_path.touch()


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--strategy', type=str, required=True,
                 choices=['sensitivity', 'in-degree', 'out-degree', 'mmse', 'jaccard',
                          'influence', 'upstream', 'entropy-diversity',
                          'infomax', 'anchor-reporter', 'anchor-sensitivity'])
  p.add_argument('--original-network-idx', type=int, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--networks-file', type=str, default=None)
  p.add_argument('--b-file', type=str, default=None)
  p.add_argument('--output-dir', type=str, required=True)
  p.add_argument('--network-size', type=int, default=5000)
  p.add_argument('--feature-sizes', type=int, nargs='+', required=True)
  p.add_argument('--num-trials', type=int, default=1)
  p.add_argument('--num-workers', type=int, default=4)
  p.add_argument('--seed', type=int, default=2025)
  p.add_argument('--beta', type=float, default=1.0,
                 help='MI redundancy penalty for entropy based strategies')
  p.add_argument('--anchor-fraction', type=float, default=0.25,
                 help='fraction of the panel picked as anchors in anchor-reporter')
  return p.parse_args()


if __name__ == '__main__':
  main(parse_args())
