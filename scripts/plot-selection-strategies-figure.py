#!/usr/bin/env python3
'''Selection strategy comparison: accuracy vs number of reporters.

Two panels (rho = 0.99 and rho = 0.5). Curves: genetic algorithm,
random selection, and the heuristic baselines (top sensitivity,
max in-degree, max out-degree, greedy MMSE, Jaccard coverage),
all scored with the same random forest evaluator.

Inputs are the per-network result CSVs produced by
genetic-algorithm-selection.py, random-node-selection.py, and
heuristic-node-selection.py.

Usage:
  python scripts/plot-selection-strategies-figure.py \
    --strategies-dir data/selection-strategies \
    --ga-csv-99 data/drug-fixed-targets-v5/N5000/ga-results-v5/combined-full.csv \
    --ga-csv-50 data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
    --random-dir-99 data/drug-fixed-targets-v5/N5000/random-results-v5 \
    --random-dir-50 data/drug-fixed-targets-v7/N5000/random-results-v7 \
    --out-dir plots/fig-strategies
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANCE = 1 / 11

STYLE = {
  'genetic':     {'color': '#2ca02c', 'label': 'genetic algorithm', 'lw': 2.6, 'zorder': 10},
  'sensitivity': {'color': '#d62728', 'label': 'most sensitive', 'lw': 2.0, 'zorder': 8},
  'in-degree':   {'color': '#ff7f0e', 'label': 'max in-degree', 'lw': 1.8, 'zorder': 6},
  'out-degree':  {'color': '#8c564b', 'label': 'max out-degree', 'lw': 1.8, 'zorder': 5},
  'mmse':        {'color': '#9467bd', 'label': 'greedy MMSE', 'lw': 1.8, 'zorder': 7},
  'jaccard':     {'color': '#e377c2', 'label': 'Jaccard coverage', 'lw': 1.8, 'zorder': 4},
  'random':      {'color': '#7f7f7f', 'label': 'random', 'lw': 2.0, 'zorder': 3},
}

plt.rcParams.update({
  'font.size': 19,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def agg(df):
  per_net = df.groupby(['max_num_features', 'original_network_idx'])['accuracy'].mean()
  g = per_net.groupby('max_num_features')
  return g.mean(), g.sem()


def load_dir(results_dir):
  frames = [pd.read_csv(p) for p in sorted(pathlib.Path(results_dir).glob('*-full.csv'))
            if p.name != 'combined-full.csv']
  return pd.concat(frames, ignore_index=True)


def load_ga(ga_csv):
  ga = pd.read_csv(ga_csv)
  final = ga.loc[ga.groupby(['original_network_idx', 'max_num_features'])['generation'].idxmax()]
  df = final.rename(columns={'best_accuracy': 'accuracy'})
  full = df.pivot(index='original_network_idx', columns='max_num_features', values='accuracy')
  # a missing cell means the GA stopped early after reaching perfect accuracy
  full = full.fillna(1.0)
  return full.stack().rename('accuracy').reset_index()


def draw_panel(ax, rho, strategies_dir, ga_csv, random_dir, show_legend):
  curves = {'genetic': load_ga(ga_csv), 'random': load_dir(random_dir)}
  for s in ['sensitivity', 'in-degree', 'out-degree', 'mmse', 'jaccard']:
    curves[s] = load_dir(pathlib.Path(strategies_dir) / f'rho{rho}' / f'{s}-results')

  for name, df in curves.items():
    mean, sem = agg(df)
    st = STYLE[name]
    x = mean.index.to_numpy()
    ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=st['color'], alpha=0.18, lw=0, zorder=st['zorder'] - 1)
    ax.plot(x, mean.to_numpy(), color=st['color'], lw=st['lw'],
            label=st['label'], zorder=st['zorder'], solid_capstyle='round')

  ax.axhline(CHANCE, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)), zorder=1)
  ax.text(2.1, CHANCE + 0.017, 'chance', fontsize=15, color='#999999')
  ax.set_xscale('log', base=2)
  ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
  ax.set_xticklabels(['1', '4', '16', '64', '256', '1024', '4096'])
  ax.set_xlim(0.9, 5600)
  ax.set_ylim(0, 1.02)
  ax.set_xlabel('Number of reporters, $m$')
  ax.set_title(f'$\\varepsilon = {2 * (1 - float(rho)):g}$', fontsize=20)
  if show_legend:
    ax.legend(frameon=False, fontsize=16, loc='lower right', handlelength=1.6)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--strategies-dir', type=str, required=True)
  p.add_argument('--ga-csv-99', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--random-dir-99', type=str, required=True)
  p.add_argument('--random-dir-50', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
  draw_panel(axes[0], 0.99, args.strategies_dir, args.ga_csv_99, args.random_dir_99, True)
  draw_panel(axes[1], 0.5, args.strategies_dir, args.ga_csv_50, args.random_dir_50, False)
  axes[0].set_ylabel('Classification accuracy')
  for ax, letter in zip(axes, 'ab'):
    ax.text(-0.10 if letter == 'a' else -0.045, 1.05, letter, transform=ax.transAxes,
            fontsize=24, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-strategies.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-strategies.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-strategies.svg + .png')


if __name__ == '__main__':
  main()
