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


def panel_a(data_dir, out_dir):
  fig, ax = plt.subplots(figsize=(7.4, 5.0))
  ymax = 0.225
  gc_ref = gamma_c(max(NS))

  ax.axvspan(1.5, gc_ref, color='#fbe9e3', zorder=0)
  ax.axvspan(gc_ref, 2.8, color='#e9eff7', zorder=0)
  ax.text((1.5 + gc_ref) / 2, 0.2135, 'chaotic', color='#b2543f',
          fontsize=14, style='italic', ha='center')
  ax.text((gc_ref + 2.8) / 2, 0.2135, 'frozen', color='#41618c',
          fontsize=14, style='italic', ha='center')
  ax.annotate(
    f'$\\gamma_c(N{{=}}{max(NS)})$', xy=(gc_ref + 0.015, 0.1665),
    color=COLORS[max(NS)], fontsize=11, ha='left',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9),
  )

  label_y = {50: 0.027, 250: 0.062, 500: 0.112, 5000: 0.196}
  for n in NS:
    mean, sem = load_curve(data_dir, n)
    x = mean.index.to_numpy()
    m = mean.to_numpy()
    s = sem.to_numpy()
    ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=COLORS[n], alpha=0.25, linewidth=0)
    ax.plot(x, m, color=COLORS[n], linewidth=2.0, solid_capstyle='round')
    gc = gamma_c(n)
    ax.axvline(gc, ymax=0.92, color=COLORS[n], linewidth=1.3 if n == max(NS) else 1.0,
               linestyle=(0, (4, 3)), alpha=0.9 if n == max(NS) else 0.75)
    ax.annotate(
      f'$N = {n}$', xy=(1.515, label_y[n]),
      color=COLORS[n], fontsize=12, fontweight='bold', ha='left',
      bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.9),
      zorder=10,
    )
  ax.text(1.985, 0.1275, 'theory $\\gamma_c(N)$', color='#555555', fontsize=10.5,
          ha='center', rotation=90)
  ax.set_ylim(0, ymax)
  ax.set_xlim(1.5, 2.8)
  ax.set_xlabel('Degree exponent, $\\gamma$')
  ax.set_ylabel('Steady state Hamming distance')
  save(fig, out_dir, 'fig2a-phase-transition')


def panel_b(degree_file, out_dir, norm=1):
  df = pd.read_csv(degree_file)
  out = df[df['kind'] == 'out'].copy()
  out['count'] = out['count'] / norm
  out = out[out['count'] > 0]
  fig, ax = plt.subplots(figsize=(3.4, 2.9))
  ax.scatter(out['degree'], out['count'], s=14, color='#1c5cab', alpha=0.85, edgecolors='none')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('Out-degree, $k$')
  ax.set_ylabel('Count')
  save(fig, out_dir, 'fig2b-out-degree')


def panel_c(degree_file, out_dir, norm=1):
  df = pd.read_csv(degree_file)
  ind = df[df['kind'] == 'in'].copy()
  ind['count'] = ind['count'] / norm
  ind = ind[ind['degree'] <= 30]
  fig, ax = plt.subplots(figsize=(3.4, 2.9))
  ax.bar(ind['degree'], ind['count'], width=0.9, color='#3987e5', edgecolor='white', linewidth=0.5)
  ax.set_xlim(-0.5, ind['degree'].max() + 0.5)
  ax.set_xlabel('In-degree, $k$')
  ax.set_ylabel('Count')
  save(fig, out_dir, 'fig2c-in-degree')


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


if __name__ == '__main__':
  main()
