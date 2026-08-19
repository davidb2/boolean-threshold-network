#!/usr/bin/env python3
'''Panel size figure: why the analysis focuses on m = 8.

  a  classification accuracy vs panel size m for evolved and random
     panels at noise eps = 0, 0.5, 1; m = 8 is the smallest size at
     which evolved panels are near their ceiling at every noise level
  b  highly sensitive fraction of evolved panels vs m, against the
     hypergeometric random expectation
  c  mean sensitivity of selected vs non-selected nodes vs m

Uses the full panel-size GA sweeps at eps = 0, 0.5, 1.

Usage:
  python scripts/plot-panel-size-figure.py \
    --ga-csvs ... ... ... \
    --random-dirs ... ... ... \
    --b-files ... ... ... \
    --eps-labels 0 0.5 1 \
    --out-dir plots/fig-panel-size
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KS = [1, 2, 4, 8, 16, 32, 64, 128]
# evolved greens dark to light with increasing noise, randoms likewise gray
GA_C = ['#14571a', '#2ca02c', '#98df8a']
RND_C = ['#4a4a4a', '#7f7f7f', '#c7c7c7']
CHANCE = 1 / 11

plt.rcParams.update({
  'font.size': 21,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def load_ga(ga_csv):
  ga = pd.read_csv(ga_csv)
  fin = ga.loc[ga.groupby(['original_network_idx', 'max_num_features'])['generation'].idxmax()]
  fin = fin[fin.max_num_features.isin(KS)]
  return fin


def load_random(random_dir):
  frames = [pd.read_csv(p) for p in sorted(pathlib.Path(random_dir).glob('*-full.csv'))]
  df = pd.concat(frames, ignore_index=True)
  return df[df.max_num_features.isin(KS)]


def acc_line(ax, df, col, color, label, ls='-'):
  per_net = df.groupby(['max_num_features', 'original_network_idx'])[col].mean()
  g = per_net.groupby('max_num_features')
  m, s = g.mean(), g.sem()
  x = m.index.to_numpy()
  ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=color, alpha=0.18, lw=0)
  ax.plot(x, m.to_numpy(), color=color, lw=2.0, label=label, linestyle=ls,
          marker='o', markersize=4)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ga-csvs', type=str, nargs=3, required=True)
  p.add_argument('--random-dirs', type=str, nargs=3, required=True)
  p.add_argument('--b-files', type=str, nargs=3, required=True)
  p.add_argument('--eps-labels', type=str, nargs=3, required=True)
  p.add_argument('--s-file', type=str, default=None,
                 help='per shock deviations at the highest noise level, for panel d')
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  cohorts = []
  for i, (ga_csv, rnd_dir, b_file, eps) in enumerate(zip(
      args.ga_csvs, args.random_dirs, args.b_files, args.eps_labels)):
    b = np.load(b_file)
    d = dict(
      eps=eps, ga=load_ga(ga_csv), rnd=load_random(rnd_dir),
      B=b['B'], bnets=[int(x) for x in b['networks']],
      ga_c=GA_C[i], rnd_c=RND_C[i],
    )
    d['cut'] = antimode(d['B'])
    cohorts.append(d)

  fig = plt.figure(figsize=(13.8, 9.4))
  gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.36)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_d = fig.add_subplot(gs[0, 1])
  ax_b = fig.add_subplot(gs[1, 0])
  ax_c = fig.add_subplot(gs[1, 1])
  axes = np.array([ax_a, ax_b, ax_c])

  for d in cohorts:
    ga = d['ga'].rename(columns={'best_accuracy': 'accuracy'})
    acc_line(ax_a, ga, 'accuracy', d['ga_c'], f'evolved, $\\varepsilon = {d["eps"]}$')
    acc_line(ax_a, d['rnd'], 'accuracy', d['rnd_c'], f'random, $\\varepsilon = {d["eps"]}$', ls=(0, (4, 2)))
  ax_a.axhline(CHANCE, color='#cccccc', lw=1.0, linestyle=(0, (3, 3)))
  ax_a.text(1.05, CHANCE - 0.055, 'chance', fontsize=15, color='#999999')
  ax_a.axvline(8, color='#e8a000', lw=1.4, alpha=0.7, linestyle=(0, (4, 2)))
  ax_a.text(8 * 1.12, 1.035, '$m = 8$', fontsize=18, color='#e8a000')
  ax_a.set_xscale('log', base=2)
  ax_a.set_xticks(KS)
  ax_a.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
  ax_a.set_xlabel('Panel size, $m$')
  ax_a.set_ylabel('Classification accuracy')
  ax_a.set_ylim(0, 1.02)
  ax_a.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none',
              fontsize=13.5, loc='lower right', bbox_to_anchor=(1.045, 0.10),
              ncol=1, handlelength=1.2, labelspacing=0.3, borderaxespad=0.0)

  for i, d in enumerate(cohorts):
    # shade already encodes the noise level (legend in panel a), so panels
    # b and c carry one legend entry per family
    mid = i == 1
    rows = []
    for _, r in d['ga'].iterrows():
      nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
      bi = d['bnets'].index(int(r['original_network_idx']))
      bvals = d['B'][bi, nodes]
      rows.append(dict(
        max_num_features=r['max_num_features'],
        original_network_idx=r['original_network_idx'],
        frac_sens=float((bvals > d['cut']).mean()),
        expect=float((d['B'][bi] > d['cut']).mean()),
        sel_B=float(bvals.mean()),
        other_B=float(np.delete(d['B'][bi], nodes).mean()),
      ))
    comp = pd.DataFrame(rows)
    acc_line(ax_b, comp, 'frac_sens', d['ga_c'], 'evolved' if mid else '_nolegend_')
    acc_line(ax_b, comp, 'expect', d['rnd_c'], 'random' if mid else '_nolegend_', ls=(0, (4, 2)))
    acc_line(ax_c, comp, 'sel_B', d['ga_c'], 'selected' if mid else '_nolegend_')
    acc_line(ax_c, comp, 'other_B', d['rnd_c'], 'not selected' if mid else '_nolegend_', ls=(0, (4, 2)))

  for ax, ylab in [(ax_b, 'Highly sensitive\nfraction of panel'), (ax_c, 'Mean node sensitivity')]:
    ax.set_xscale('log', base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(['1', '2', '4', '8', '16', '', '64', ''])
    ax.set_xlabel('Panel size, $m$')
    ax.set_ylabel(ylab)
    ax.axvline(8, color='#e8a000', lw=1.4, alpha=0.7, linestyle=(0, (4, 2)))
  ax_b.set_ylim(0, 1.0)
  ax_b.legend(frameon=False, fontsize=16, loc='upper right', handlelength=1.2)
  ax_c.set_ylim(0, 0.55)
  ax_c.legend(frameon=False, fontsize=16, loc='upper right', handlelength=1.2)

  # d: effective number of shocks by member rank, at the highest noise level
  if args.s_file:
    sd = np.load(args.s_file, allow_pickle=True)
    Sp, snets = sd['S'], [int(x) for x in sd['networks']]
    dd = cohorts[2]
    eff = []
    for _, r in dd['ga'][dd['ga'].max_num_features == 8].iterrows():
      nodes = [int(s.split('-')[1]) for s in eval(r['features'])]
      if len(set(nodes)) != 8 or int(r['original_network_idx']) not in snets:
        continue
      sj = snets.index(int(r['original_network_idx']))
      bi = dd['bnets'].index(int(r['original_network_idx']))
      members = sorted(set(nodes), key=lambda n_: -dd['B'][bi, n_])
      tot = Sp[sj][:, members].sum(axis=0)
      sq = (Sp[sj][:, members] ** 2).sum(axis=0)
      with np.errstate(divide='ignore', invalid='ignore'):
        eff.append(np.where(sq > 0, tot ** 2 / sq, 0.0))
    eff = np.array(eff)
    m = eff.mean(axis=0)
    se = 1.96 * eff.std(axis=0) / np.sqrt(eff.shape[0])
    ax_d.errorbar(range(1, 9), m, yerr=se, color='#2ca02c', lw=2.0,
                  marker='o', markersize=5, capsize=3)
    ax_d.set_xlabel('Member rank by sensitivity')
    ax_d.set_ylabel('Effective number of shocks')
    ax_d.set_ylim(0, 10)
    ax_d.set_xticks(range(1, 9))
    ax_d.set_title(f'$\\varepsilon = {cohorts[2]["eps"]}$, $m = 8$', fontsize=17)

  for ax, letter, dx in [(ax_a, 'a', -0.14), (ax_d, 'b', -0.23),
                         (ax_b, 'c', -0.14), (ax_c, 'd', -0.14)]:
    ax.text(dx, 1.04, letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-panel-size.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-panel-size.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-panel-size.svg + .png')


if __name__ == '__main__':
  main()
