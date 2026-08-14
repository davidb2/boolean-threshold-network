#!/usr/bin/env python3
'''Supplementary figures.

  si-activity      control-run activity vs shock sensitivity B
  si-redundancy    within-panel information: marginal entropy and pairwise MI
  si-qk            Monte Carlo check of the single-node sensitivity q(K)
  si-bdist         sensitivity distributions and antimode cutoffs across noise
  si-convergence   genetic algorithm convergence at m = 8 across noise

Usage:
  python scripts/plot-si-figures.py --sensitivity-dir data/sensitivity \
    --sweep-dir data/drug-rho-sweep --ga-csv-99 ... --ga-csv-50 ... --out-dir plots/si
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RHOS = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85',
        '0.9', '0.925', '0.95', '0.975', '0.99', '0.995']

plt.rcParams.update({
  'font.size': 11.5,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def save(fig, out_dir, name):
  out_dir = pathlib.Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png')
  plt.close(fig)


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def si_activity(sens_dir, out_dir):
  fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
  for ax, rho in zip(axes, ['0.99', '0.5']):
    a = np.load(f'{sens_dir}/activity-rho{rho}.npz')
    act, anets = a['activity'], [int(x) for x in a['networks']]
    b = np.load(f'{sens_dir}/B-rho{rho}.npz')
    B, bnets = b['B'], [int(x) for x in b['networks']]
    A = act[[anets.index(n) for n in bnets]]
    hb = ax.hexbin(A.ravel(), B.ravel(), gridsize=45, cmap='Blues',
                   mincnt=1, bins='log')
    r = np.corrcoef(A.ravel(), B.ravel())[0, 1]
    ax.set_xlabel('Control state variance')
    ax.set_ylabel('Sensitivity, $B$')
    ax.set_title(f'$\\rho = {rho}$, $r = {r:.2f}$', fontsize=12)
  fig.colorbar(hb, ax=axes, label='nodes (log scale)', shrink=0.85)
  save(fig, out_dir, 'si-activity')


def si_redundancy(sens_dir, out_dir):
  groups = [('evolved', '#0f3560', 'evolved'),
            ('matched', '#eb6834', 'random, sensitivity matched'),
            ('random', '#8b93a1', 'random')]
  cols = [('mean_marg_H', 'Marginal entropy\nper member (bits)'),
          ('mean_mi', 'Mean pairwise\nmutual information (bits)'),
          ('max_mi', 'Largest pairwise\nmutual information (bits)')]
  fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
  for ax, (col, ylab) in zip(axes, cols):
    for gi, (group, color, label) in enumerate(groups):
      xs, ms, ss = [], [], []
      for xi, rho in enumerate(['0.99', '0.5']):
        df = pd.read_csv(f'{sens_dir}/redundancy-rho{rho}.csv')
        df['group'] = df['panel'].str.replace(r'-\d+$', '', regex=True)
        per_net = df[df.group == group].groupby('network')[col].mean()
        xs.append(xi + (gi - 1) * 0.24)
        ms.append(per_net.mean())
        ss.append(1.96 * per_net.sem())
      ax.bar(xs, ms, width=0.21, color=color, label=label, zorder=3)
      ax.errorbar(xs, ms, yerr=ss, fmt='none', ecolor='#333333', lw=1.0, capsize=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$\\rho = 0.99$', '$\\rho = 0.5$'])
    ax.set_ylabel(ylab)
  axes[0].legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0.0, 1.18))
  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.24, 1.05, letter, transform=ax.transAxes, fontsize=15,
            fontweight='bold', color='#222222')
  save(fig, out_dir, 'si-redundancy')


def si_qk(out_dir):
  rng = np.random.default_rng(7)
  M = 500_000
  c = 0.5 / (np.sqrt(1 / 3) * np.sqrt(np.pi))
  ks = np.array([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200])
  qs = []
  for K in ks:
    w = rng.uniform(-1, 1, size=(M, K))
    s = (rng.random(size=(M, K)) < 0.5).astype(np.float64)
    prev = (rng.random(M) < 0.5).astype(np.float64)
    def out(states):
      f = (w * states).sum(axis=1)
      return np.where(f > 0, 1.0, np.where(f < 0, 0.0, prev))
    base = out(s)
    s2 = s.copy()
    idx = rng.integers(0, K, size=M)
    s2[np.arange(M), idx] = 1 - s2[np.arange(M), idx]
    qs.append((base != out(s2)).mean())
  fig, ax = plt.subplots(figsize=(5.4, 4.0))
  kk = np.linspace(1, 220, 300)
  ax.plot(kk, c / np.sqrt(kk), color='#8b93a1', lw=1.6,
          label='$q(K) = \\sqrt{3/(4\\pi K)}$')
  ax.plot(ks, qs, 'o', color='#0f3560', markersize=6, label='Monte Carlo')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('In-degree, $K$')
  ax.set_ylabel('Single input sensitivity, $q(K)$')
  ax.legend(frameon=False, fontsize=10)
  save(fig, out_dir, 'si-qk')


def si_bdist(sens_dir, out_dir):
  fig, axes = plt.subplots(2, 7, figsize=(15.5, 4.6), sharex=True, sharey=True)
  for ax, rho in zip(axes.ravel(), RHOS):
    b = np.load(f'{sens_dir}/B-rho{rho}.npz')
    B = b['B'].ravel()
    cut = antimode(b['B'])
    ax.hist(B, bins=60, range=(0, 0.8), color='#3987e5', alpha=0.85)
    ax.axvline(cut, color='#e8a000', lw=1.4, linestyle=(0, (4, 2)))
    ax.set_title(f'$\\rho = {rho}$', fontsize=10.5)
    ax.set_yscale('log')
  for ax in axes[1]:
    ax.set_xlabel('$B$', fontsize=10)
  for ax in axes[:, 0]:
    ax.set_ylabel('Nodes')
  save(fig, out_dir, 'si-bdist')


def si_convergence(sweep_dir, ga_csv_99, ga_csv_50, out_dir):
  fig, ax = plt.subplots(figsize=(6.4, 4.2))
  cmap = plt.get_cmap('Blues')
  for i, rho in enumerate(RHOS):
    if rho == '0.99':
      path = ga_csv_99
    elif rho == '0.5':
      path = ga_csv_50
    else:
      path = f'{sweep_dir}/rho{rho}/ga-results/combined-full.csv'
    ga = pd.read_csv(path)
    ga = ga[ga.max_num_features == 8]
    m = ga.groupby('generation')['best_accuracy'].mean()
    ax.plot(m.index.to_numpy(), m.to_numpy(),
            color=cmap(0.35 + 0.6 * i / (len(RHOS) - 1)), lw=1.5)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, 0.995))
  fig.colorbar(sm, ax=ax, label='$\\rho$', shrink=0.85)
  ax.set_xlabel('Generation')
  ax.set_ylabel('Best panel accuracy ($m = 8$)')
  ax.set_ylim(0.5, 1.02)
  save(fig, out_dir, 'si-convergence')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--sweep-dir', type=str, required=True)
  p.add_argument('--ga-csv-99', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()
  si_activity(args.sensitivity_dir, args.out_dir)
  si_redundancy(args.sensitivity_dir, args.out_dir)
  si_qk(args.out_dir)
  si_bdist(args.sensitivity_dir, args.out_dir)
  si_convergence(args.sweep_dir, args.ga_csv_99, args.ga_csv_50, args.out_dir)


if __name__ == '__main__':
  main()
