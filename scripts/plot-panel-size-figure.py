#!/usr/bin/env python3
'''Panel size figure: why the analysis focuses on m = 8.

  a  classification accuracy vs panel size m for evolved and random
     panels at low and high noise; m = 8 is the smallest size at which
     evolved panels are near their ceiling at both noise levels
  b  number of sensitive nodes in evolved panels vs m, against the
     hypergeometric random expectation
  c  mean sensitivity of selected vs non-selected nodes vs m

Uses the full panel-size GA sweeps at rho = 0.99 and rho = 0.5.

Usage:
  python scripts/plot-panel-size-figure.py \
    --ga-csv-99 ... --ga-csv-50 ... \
    --random-dir-99 ... --random-dir-50 ... \
    --b-file-99 data/sensitivity/B-rho0.99.npz \
    --b-file-50 data/sensitivity/B-rho0.5.npz \
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
EPS = {'0.99': '0.02', '0.5': '1'}
GA_C = {'0.99': '#0f3560', '0.5': '#3987e5'}
RND_C = {'0.99': '#6f7885', '0.5': '#a8b0bb'}
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
  p.add_argument('--ga-csv-99', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--random-dir-99', type=str, required=True)
  p.add_argument('--random-dir-50', type=str, required=True)
  p.add_argument('--b-file-99', type=str, required=True)
  p.add_argument('--b-file-50', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  data = {}
  for rho, ga_csv, rnd_dir, b_file in [
    ('0.99', args.ga_csv_99, args.random_dir_99, args.b_file_99),
    ('0.5', args.ga_csv_50, args.random_dir_50, args.b_file_50),
  ]:
    b = np.load(b_file)
    data[rho] = dict(
      ga=load_ga(ga_csv), rnd=load_random(rnd_dir),
      B=b['B'], bnets=[int(x) for x in b['networks']],
    )
    data[rho]['cut'] = antimode(data[rho]['B'])

  fig = plt.figure(figsize=(13.2, 9.4))
  gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.32)
  ax_a = fig.add_subplot(gs[0, :])
  ax_b = fig.add_subplot(gs[1, 0])
  ax_c = fig.add_subplot(gs[1, 1])
  axes = np.array([ax_a, ax_b, ax_c])

  for rho in ['0.99', '0.5']:
    d = data[rho]
    ga = d['ga'].rename(columns={'best_accuracy': 'accuracy'})
    acc_line(ax_a, ga, 'accuracy', GA_C[rho], f'evolved, $\\varepsilon = {EPS[rho]}$')
    acc_line(ax_a, d['rnd'], 'accuracy', RND_C[rho], f'random, $\\varepsilon = {EPS[rho]}$', ls=(0, (4, 2)))
  ax_a.axhline(CHANCE, color='#cccccc', lw=1.0, linestyle=(0, (3, 3)))
  ax_a.text(1.3, CHANCE + 0.02, 'chance', fontsize=16, color='#999999')
  ax_a.axvline(8, color='#e8a000', lw=1.4, alpha=0.7, linestyle=(0, (4, 2)))
  ax_a.text(8 * 1.12, 1.005, '$m = 8$', fontsize=18, color='#e8a000')
  ax_a.set_xscale('log', base=2)
  ax_a.set_xticks(KS)
  ax_a.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
  ax_a.set_xlabel('Panel size, $m$')
  ax_a.set_ylabel('Classification accuracy')
  ax_a.set_ylim(0, 1.02)
  ax_a.legend(frameon=False, fontsize=18, loc='center right')

  for rho in ['0.99', '0.5']:
    d = data[rho]
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
    acc_line(ax_b, comp, 'frac_sens', GA_C[rho], f'evolved, $\\varepsilon = {EPS[rho]}$')
    acc_line(ax_b, comp, 'expect', RND_C[rho], f'random, $\\varepsilon = {EPS[rho]}$', ls=(0, (4, 2)))
    acc_line(ax_c, comp, 'sel_B', GA_C[rho], f'selected, $\\varepsilon = {EPS[rho]}$')
    acc_line(ax_c, comp, 'other_B', RND_C[rho], f'not selected, $\\varepsilon = {EPS[rho]}$', ls=(0, (4, 2)))

  for ax, ylab in [(ax_b, 'Sensitive fraction of panel'), (ax_c, 'Mean node sensitivity')]:
    ax.set_xscale('log', base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(['1', '2', '4', '8', '16', '', '64', ''])
    ax.set_xlabel('Panel size, $m$')
    ax.set_ylabel(ylab)
    ax.axvline(8, color='#e8a000', lw=1.4, alpha=0.7, linestyle=(0, (4, 2)))
  ax_b.set_ylim(0, 1.0)
  ax_b.legend(frameon=False, fontsize=15, loc='upper right', handlelength=1.0)
  ax_c.set_ylim(0, 0.55)
  ax_c.legend(frameon=False, fontsize=15, loc='upper right', handlelength=1.0)

  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-panel-size.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-panel-size.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-panel-size.svg + .png')


if __name__ == '__main__':
  main()
