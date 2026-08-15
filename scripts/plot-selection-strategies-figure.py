#!/usr/bin/env python3
'''Selection strategy comparison: accuracy vs number of reporters.

Three panels (noise eps = 0, 0.5, 1). Foreground curves: genetic
algorithm, random selection, most sensitive, greedy information gain
(infomax), the two phase anchors plus reporters rule, and greedy
influence maximization. The remaining structural and information
heuristics (in degree, out degree, greedy MMSE, Jaccard coverage,
upstream coverage, entropy with diversity) are drawn as a thin gray
background family; their full curves are in the SI.

The entropy based strategies stop at m = 16, where the plug in
estimator they optimize is reliable.

Usage:
  python scripts/plot-selection-strategies-figure.py \
    --strategies-dirs data/selection-strategies/rho1.0 \
                      data/selection-strategies/rho0.75-b4 \
                      data/selection-strategies/rho0.5 \
    --ga-csvs ... ... ... \
    --random-dirs ... ... ... \
    --eps-labels 0 0.5 1 \
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

FOREGROUND = {
  'genetic':         {'color': '#2ca02c', 'label': 'genetic algorithm', 'lw': 3.2, 'zorder': 12},
  'random':          {'color': '#7f7f7f', 'label': 'random', 'lw': 2.4, 'zorder': 4},
  'sensitivity':     {'color': '#ff7f0e', 'label': 'most sensitive', 'lw': 2.4, 'zorder': 9},
  'infomax':         {'color': '#17becf', 'label': 'greedy information gain', 'lw': 2.4, 'zorder': 10},
  'anchor-reporter': {'color': '#9467bd', 'label': 'anchors plus reporters', 'lw': 2.4, 'zorder': 8},
  'influence':       {'color': '#8c564b', 'label': 'influence maximization', 'lw': 2.4, 'zorder': 6},
}
BACKGROUND = ['in-degree', 'out-degree', 'mmse', 'jaccard', 'upstream', 'entropy-diversity']
BG_COLOR = '#c7c7c7'

plt.rcParams.update({
  'font.size': 20,
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


def draw_panel(ax, strategies_dir, ga_csv, random_dir):
  for st in BACKGROUND:
    d = pathlib.Path(strategies_dir) / f'{st}-results'
    if not d.exists():
      continue
    mean, sem = agg(load_dir(d))
    ax.plot(mean.index.to_numpy(), mean.to_numpy(), color=BG_COLOR, lw=1.3,
            zorder=2, solid_capstyle='round')

  handles = []
  for name, st in FOREGROUND.items():
    if name == 'genetic':
      df = load_ga(ga_csv)
    elif name == 'random':
      df = load_dir(random_dir)
    else:
      d = pathlib.Path(strategies_dir) / f'{name}-results'
      if not d.exists():
        continue
      df = load_dir(d)
    mean, sem = agg(df)
    x = mean.index.to_numpy()
    ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=st['color'], alpha=0.16, lw=0, zorder=st['zorder'] - 1)
    line, = ax.plot(x, mean.to_numpy(), color=st['color'], lw=st['lw'],
                    label=st['label'], zorder=st['zorder'], solid_capstyle='round')
    handles.append(line)
  return handles


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--strategies-dirs', type=str, nargs=3, required=True)
  p.add_argument('--ga-csvs', type=str, nargs=3, required=True)
  p.add_argument('--random-dirs', type=str, nargs=3, required=True)
  p.add_argument('--eps-labels', type=str, nargs=3, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.9), sharey=True)
  fig.subplots_adjust(wspace=0.14, bottom=0.32)
  handles = []
  for ax, sdir, gcsv, rdir, eps in zip(axes, args.strategies_dirs, args.ga_csvs,
                                       args.random_dirs, args.eps_labels):
    handles = draw_panel(ax, sdir, gcsv, rdir) or handles
    ax.axhline(CHANCE, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64])
    ax.set_xticklabels(['1', '4', '16', '64'])
    ax.set_xlim(0.9, 150)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Number of reporters, $m$')
    ax.set_title(f'$\\varepsilon = {eps}$', fontsize=21)
  axes[0].set_ylabel('Classification accuracy')
  axes[0].text(1.1, CHANCE + 0.02, 'chance', fontsize=14, color='#999999')

  bg_proxy = plt.Line2D([], [], color=BG_COLOR, lw=1.3, label='other heuristics')
  fig.legend(handles=handles + [bg_proxy], loc='lower center', frameon=False,
             fontsize=17, ncol=4, bbox_to_anchor=(0.5, 0.015), columnspacing=1.4)
  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.06, 1.05, letter, transform=ax.transAxes,
            fontsize=27, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-strategies.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-strategies.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-strategies.svg + .png')


if __name__ == '__main__':
  main()
