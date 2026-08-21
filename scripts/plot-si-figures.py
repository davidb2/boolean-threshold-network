#!/usr/bin/env python3
'''Supplementary figures.

  si-activity      control-run activity vs shock sensitivity B
  si-redundancy    within-panel information: marginal entropy and pairwise MI
  si-qk            simulation check of the single-node sensitivity q(K)
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
  'font.size': 18,
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
    ax.set_ylabel('Sensitivity, $S$')
    ax.set_title(f'$\\varepsilon = {2 * (1 - float(rho)):g}$, $r = {r:.2f}$', fontsize=19)
  fig.colorbar(hb, ax=axes, label='nodes (log scale)', shrink=0.85)
  save(fig, out_dir, 'si-activity')


def si_redundancy(sens_dir, out_dir):
  groups = [('evolved', '#2ca02c', 'evolved'),
            ('matched', '#ff7f0e', 'random, sensitivity matched'),
            ('random', '#7f7f7f', 'random')]
  cols = [('mean_marg_H', 'Marginal entropy\nper member (bits)'),
          ('mean_mi', 'Mean pairwise\nMI (bits)'),
          ('max_mi', 'Largest pairwise\nMI (bits)')]
  fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
  fig.subplots_adjust(wspace=0.42, top=0.78)
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
    ax.set_xticklabels(['$\\varepsilon = 0.02$', '$\\varepsilon = 1$'])
    ax.set_ylabel(ylab)
  axes[1].legend(frameon=False, fontsize=16, loc='lower center',
                 bbox_to_anchor=(0.5, 1.02), ncol=3, columnspacing=1.2)
  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.24, 1.05, letter, transform=ax.transAxes, fontsize=24,
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
  fig, ax = plt.subplots(figsize=(6.0, 4.6))
  kk = np.linspace(1, 220, 300)
  ax.plot(kk, c / np.sqrt(kk), color='#7f7f7f', lw=1.6, label='Theory')
  ax.plot(ks, qs, 'o', color='#1f77b4', markersize=6, label='Simulation')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('In-degree, $K$')
  ax.set_ylabel('Single input sensitivity, $q(K)$')
  leg = ax.legend(frameon=True, fontsize=15, framealpha=1.0, borderpad=0.6,
                  edgecolor='#999999', fancybox=False)
  leg.get_frame().set_linewidth(0.8)
  save(fig, out_dir, 'si-qk')


def si_bdist(sens_dir, out_dir):
  '''Sensitivity distributions at four noise levels, on a linear count scale.

  The frozen spike at S = 0 holds about a quarter of all nodes and is
  clipped so the rest of the distribution is visible; its true height is
  printed above each panel. Linear counts make the dip between the frozen
  population and the second mode legible, which a logarithmic scale flattens.
  '''
  shown = ['1.0', '0.9', '0.75', '0.5']  # eps = 0, 0.2, 0.5, 1
  fig, axes = plt.subplots(1, len(shown), figsize=(15.0, 3.9),
                           sharex=True, sharey=True)
  for ax, rho in zip(axes, shown):
    b = np.load(f'{sens_dir}/B-rho{rho}.npz')
    B = b['B'].ravel()
    cut = antimode(b['B'])
    counts, edges = np.histogram(B, bins=60, range=(0, 0.8))
    ax.hist(B, bins=60, range=(0, 0.8), color='#7f7f7f', alpha=0.85)
    ax.axvline(cut, color='#e8a000', lw=1.6, linestyle=(0, (4, 2)))
    ax.set_title(f'$\\varepsilon = {2 * (1 - float(rho)):g}$', fontsize=17)
    ax.set_xlabel('$S$', fontsize=16)
    ceiling = counts[2:].max() * 1.45
    ax.set_ylim(0, ceiling)
    if counts[0] > ceiling:  # only annotate a spike that is actually clipped
      ax.annotate(f'{counts[0] / 1000:.0f}k, off scale',
                  xy=(edges[1], ceiling * 0.99),
                  xytext=(edges[1] + 0.06, ceiling * 0.86), fontsize=13,
                  color='#555555',
                  arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
  axes[0].set_ylabel('Nodes')
  save(fig, out_dir, 'si-bdist')


GA_GREENS = ['#14571a', '#2ca02c', '#98df8a']


def si_convergence(agg_csv, out_dir):
  d = pd.read_csv(agg_csv)
  fig, ax = plt.subplots(figsize=(6.8, 4.4))
  for i, (eps, g) in enumerate(sorted(d.groupby('eps'))):
    g = g.sort_values('generation')
    lbl = ('$\\varepsilon = %g$' % eps)
    ax.plot(g.generation, g.best, color=GA_GREENS[i], lw=2.0, label=lbl)
    ax.plot(g.generation, g.avg, color=GA_GREENS[i], lw=1.6,
            linestyle=(0, (4, 2)))
  ax.plot([], [], color='#777777', lw=2.0, label='best')
  ax.plot([], [], color='#777777', lw=1.6, linestyle=(0, (4, 2)),
          label='population average')
  ax.set_xlabel('Generation')
  ax.set_ylabel('Fitness ($m = 8$)')
  ax.set_ylim(0.1, 1.02)
  ax.legend(frameon=False, fontsize=11, loc='lower right',
            handlelength=1.6, labelspacing=0.3)
  save(fig, out_dir, 'si-convergence')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--sweep-dir', type=str, required=True)
  p.add_argument('--ga-csv-99', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--convergence-agg-csv', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()
  si_activity(args.sensitivity_dir, args.out_dir)
  si_redundancy(args.sensitivity_dir, args.out_dir)
  si_qk(args.out_dir)
  si_bdist(args.sensitivity_dir, args.out_dir)
  si_convergence(args.convergence_agg_csv, args.out_dir)


if __name__ == '__main__':
  main()
