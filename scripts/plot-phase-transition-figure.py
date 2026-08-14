#!/usr/bin/env python3
'''Figure 2: steady state Hamming distance vs degree exponent gamma.

Panel a: damage curves for N in {50, 250, 500, 5000} with the annealed
theory prediction gamma_c(N) marked (Kbar(gamma_c, N) = 4pi/3).
Panels b, c: out-degree and in-degree distributions of one simulated
network (histogram CSV produced separately).

Outputs .svg and .png (300 dpi) per panel, for assembly in Illustrator.

Usage:
  python scripts/plot-phase-transition-figure.py \
    --data-dir data/no-drug-power-law-phase-transition \
    --degree-file plots/fig2/degree-hist.csv \
    --out-dir plots/fig2
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

NS = [50, 250, 500, 5000]
COLORS = {50: '#86b6ef', 250: '#3987e5', 500: '#1c5cab', 5000: '#0f3560'}
K_C = 4 * np.pi / 3

plt.rcParams.update({
  'font.size': 13,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'xtick.direction': 'out',
  'ytick.direction': 'out',
  'svg.fonttype': 'none',
})


def kbar(gamma, n):
  ks = np.arange(1, n + 1, dtype=np.float64)
  w = ks ** (-gamma)
  return (w * ks).sum() / w.sum()


def gamma_c(n):
  return optimize.brentq(lambda g: kbar(g, n) - K_C, 1.01, 5.0)


def load_curve(data_dir, n):
  path = next(pathlib.Path(data_dir).glob(f'N{n}-v2/derived/hamming-distances-*.csv'), None)
  if path is None:
    path = pathlib.Path(data_dir) / f'N{n}.csv'
  df = pd.read_csv(path)
  per_net = (
    df.groupby(['gamma', 'actual_connectivity'])['hamming_distance']
    .mean()
    .reset_index()
  )
  g = per_net.groupby('gamma')['hamming_distance']
  return g.mean(), g.sem()


def save(fig, out_dir, name):
  out_dir = pathlib.Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png')
  plt.close(fig)


WARM, COOL = '#fbe9e3', '#e9eff7'
WARM_TXT, COOL_TXT = '#b2543f', '#41618c'


def gamma_c_inf():
  from scipy.special import zeta as rzeta
  return optimize.brentq(lambda g: rzeta(g - 1) / rzeta(g) - K_C, 2.01, 4.0)


def gamma_c_curve():
  ns = np.unique(np.logspace(1.2, 6, 40).astype(int))
  return ns, np.array([
    optimize.brentq(
      lambda g, n=n: kbar(g, n) - K_C, 1.01, 5.0,
    ) for n in ns
  ])


def draw_panel_a(ax, data_dir):
  ymax = 0.225

  label_y = {50: 0.027, 250: 0.062, 500: 0.112, 5000: 0.196}
  for n in NS:
    mean, sem = load_curve(data_dir, n)
    x = mean.index.to_numpy()
    m = mean.to_numpy()
    s = sem.to_numpy()
    ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=COLORS[n], alpha=0.25, linewidth=0)
    ax.plot(x, m, color=COLORS[n], linewidth=2.0, solid_capstyle='round')
    gc = gamma_c(n)
    ax.plot(gc, np.interp(gc, x, m), marker='*', markersize=17,
            markerfacecolor='white', markeredgecolor=COLORS[n],
            markeredgewidth=1.5, linestyle='none', zorder=11, clip_on=False)
    ax.annotate(
      f'$N = {n}$', xy=(1.515, label_y[n]),
      color=COLORS[n], fontsize=12, fontweight='bold', ha='left',
      bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.9),
      zorder=10,
    )
  ax.annotate('', xy=(1.53, 0.2135), xytext=(1.72, 0.2135),
              arrowprops=dict(arrowstyle='-|>', lw=1.4, color=WARM_TXT))
  ax.text(1.74, 0.2135, 'more chaotic', color=WARM_TXT, fontsize=12.5,
          style='italic', va='center', ha='left')
  ax.annotate('', xy=(2.77, 0.2135), xytext=(2.58, 0.2135),
              arrowprops=dict(arrowstyle='-|>', lw=1.4, color=COOL_TXT))
  ax.text(2.56, 0.2135, 'more frozen', color=COOL_TXT, fontsize=12.5,
          style='italic', va='center', ha='right')
  ax.plot(2.315, 0.047, marker='*', markersize=15, markerfacecolor='white',
          markeredgecolor='#555555', markeredgewidth=1.3, linestyle='none', clip_on=False)
  ax.text(2.355, 0.047, 'theory $\\gamma_c(N)$', color='#555555', fontsize=10.5,
          ha='left', va='center')

  axi = ax.inset_axes([0.565, 0.335, 0.415, 0.52])
  ns, gcs = gamma_c_curve()
  gc_inf = gamma_c_inf()
  axi.fill_betweenx(ns, 1.5, gcs, color=WARM, zorder=0)
  axi.fill_betweenx(ns, gcs, 2.8, color=COOL, zorder=0)
  axi.plot(gcs, ns, color='#333333', lw=1.8)
  axi.axvline(gc_inf, color='#666666', lw=1.0, linestyle=(0, (2, 2)))
  for n in NS:
    axi.plot(gamma_c(n), n, 'o', color=COLORS[n], markersize=6.5,
             markeredgecolor='white', markeredgewidth=1.0, zorder=5)
  axi.set_yscale('log')
  axi.set_xlim(1.5, 2.8)
  axi.set_ylim(16, 1e6)
  axi.set_xlabel('$\\gamma$', fontsize=10, labelpad=1)
  axi.set_ylabel('$N$', fontsize=10, labelpad=-1)
  axi.set_xticks([1.6, 2.0, 2.4, 2.8])
  axi.tick_params(labelsize=8.5)
  axi.text(1.62, 3e4, 'chaotic', color=WARM_TXT, fontsize=10.5, style='italic')
  axi.text(2.42, 1.5e2, 'frozen', color=COOL_TXT, fontsize=10.5, style='italic')
  axi.text(2.03, 1.4e5, '$\\gamma_c(N)$', color='#333333', fontsize=10, rotation=74)
  axi.text(gc_inf + 0.035, 28, '$N \\to \\infty$', color='#666666', fontsize=9, ha='left')
  for s in axi.spines.values():
    s.set_color('#aaaaaa')

  ax.set_ylim(0, ymax)
  ax.set_xlim(1.5, 2.8)
  ax.set_xlabel('Degree exponent, $\\gamma$')
  ax.set_ylabel('Steady state Hamming distance')


def panel_a(data_dir, out_dir):
  fig, ax = plt.subplots(figsize=(7.4, 5.0))
  draw_panel_a(ax, data_dir)
  save(fig, out_dir, 'fig2a-phase-transition')


def draw_panel_b(ax, degree_file, norm=1):
  df = pd.read_csv(degree_file)
  out = df[df['kind'] == 'out'].copy()
  out['count'] = out['count'] / norm
  out = out[out['count'] > 0]
  ax.scatter(out['degree'], out['count'], s=14, color='#1c5cab', alpha=0.85, edgecolors='none')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('Out-degree, $k$')
  ax.set_ylabel('Count per network')


def panel_b(degree_file, out_dir, norm=1):
  fig, ax = plt.subplots(figsize=(3.4, 2.9))
  draw_panel_b(ax, degree_file, norm)
  save(fig, out_dir, 'fig2b-out-degree')


def draw_panel_c(ax, degree_file, norm=1):
  df = pd.read_csv(degree_file)
  ind = df[df['kind'] == 'in'].copy()
  ind['count'] = ind['count'] / norm
  ind = ind[ind['degree'] <= 30]
  ax.bar(ind['degree'], ind['count'], width=0.9, color='#3987e5', edgecolor='white', linewidth=0.5)
  ax.set_xlim(-0.5, ind['degree'].max() + 0.5)
  ax.set_xlabel('In-degree, $k$')
  ax.set_ylabel('Count per network')


def panel_c(degree_file, out_dir, norm=1):
  fig, ax = plt.subplots(figsize=(3.4, 2.9))
  draw_panel_c(ax, degree_file, norm)
  save(fig, out_dir, 'fig2c-in-degree')


def combined(data_dir, degree_file, norm, out_dir):
  fig = plt.figure(figsize=(12.2, 5.4))
  gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], hspace=0.52, wspace=0.24)
  ax_a = fig.add_subplot(gs[:, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  ax_c = fig.add_subplot(gs[1, 1])
  draw_panel_a(ax_a, data_dir)
  draw_panel_b(ax_b, degree_file, norm)
  draw_panel_c(ax_c, degree_file, norm)
  for ax, letter, dx in [(ax_a, 'a', -0.085), (ax_b, 'b', -0.28), (ax_c, 'c', -0.28)]:
    ax.text(dx, 1.04, letter, transform=ax.transAxes,
            fontsize=17, fontweight='bold', color='#222222')
  save(fig, out_dir, 'fig2-combined')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data-dir', type=str, required=True)
  p.add_argument('--degree-file', type=str, default=None)
  p.add_argument('--degree-norm', type=float, default=1)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()
  for n in NS:
    print(f'gamma_c({n}) = {gamma_c(n):.4f}')
  panel_a(args.data_dir, args.out_dir)
  if args.degree_file and pathlib.Path(args.degree_file).exists():
    panel_b(args.degree_file, args.out_dir, args.degree_norm)
    panel_c(args.degree_file, args.out_dir, args.degree_norm)
    combined(args.data_dir, args.degree_file, args.degree_norm, args.out_dir)


if __name__ == '__main__':
  main()
