#!/usr/bin/env python3
'''Full selection strategy curves (SI companion to the main strategies figure).

Same three panels (noise eps = 0, 0.5, 1) but every heuristic is drawn
with its own color and listed in the legend, including the ones the main
figure compresses into a gray background family.

The entropy based strategies (greedy information gain, anchors plus
detectors, entropy with diversity, greedy MMSE) stop at m = 16, where the
plug in estimator they optimize is reliable.

Usage: same arguments as plot-selection-strategies-figure.py.
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

CHANCE = 1 / 11

# Solid lines: the main text foreground. Dashed lines: the SI only family.
STYLES = {
  'genetic':            {'color': '#2ca02c', 'label': 'genetic algorithm', 'lw': 3.0, 'ls': '-', 'zorder': 14},
  'infomax':            {'color': '#17becf', 'label': 'greedy information gain', 'lw': 2.2, 'ls': '-', 'zorder': 12},
  'anchor-reporter':    {'color': '#9467bd', 'label': 'anchors plus detectors', 'lw': 2.2, 'ls': '-', 'zorder': 11},
  'sensitivity':        {'color': '#ff7f0e', 'label': 'most sensitive', 'lw': 2.2, 'ls': '-', 'zorder': 10},
  'influence':          {'color': '#8c564b', 'label': 'influence maximization', 'lw': 2.2, 'ls': '-', 'zorder': 6},
  'random':             {'color': '#7f7f7f', 'label': 'random', 'lw': 2.2, 'ls': '-', 'zorder': 4},
  'entropy-diversity':  {'color': '#bcbd22', 'label': 'entropy with diversity', 'lw': 1.8, 'ls': '--', 'zorder': 9},
  'anchor-sensitivity': {'color': '#e377c2', 'label': 'anchors by sensitivity only', 'lw': 1.8, 'ls': '--', 'zorder': 8},
  'mmse':               {'color': '#9edae5', 'label': 'greedy MMSE', 'lw': 1.8, 'ls': '--', 'zorder': 7},
  'upstream':           {'color': '#c49c94', 'label': 'upstream coverage', 'lw': 1.8, 'ls': '--', 'zorder': 5},
  'jaccard':            {'color': '#c5b0d5', 'label': 'Jaccard coverage', 'lw': 1.8, 'ls': '--', 'zorder': 5},
  'in-degree':          {'color': '#dbdb8d', 'label': 'in degree', 'lw': 1.8, 'ls': '--', 'zorder': 3},
  'out-degree':         {'color': '#c7c7c7', 'label': 'out degree', 'lw': 1.8, 'ls': '--', 'zorder': 3},
}

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
  handles = []
  for name, st in STYLES.items():
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
                    color=st['color'], alpha=0.12, lw=0, zorder=st['zorder'] - 1)
    line, = ax.plot(x, mean.to_numpy(), color=st['color'], lw=st['lw'], ls=st['ls'],
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

  fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.6), sharey=True)
  fig.subplots_adjust(wspace=0.14, bottom=0.40)
  handles = []
  for ax, sdir, gcsv, rdir, eps in zip(axes, args.strategies_dirs, args.ga_csvs,
                                       args.random_dirs, args.eps_labels):
    h = draw_panel(ax, sdir, gcsv, rdir)
    handles = h if len(h) > len(handles) else handles
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

  fig.legend(handles=handles, loc='lower center', frameon=False,
             fontsize=15, ncol=4, bbox_to_anchor=(0.5, 0.01), columnspacing=1.2)
  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.06, 1.05, letter, transform=ax.transAxes,
            fontsize=27, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'si-strategies-full.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'si-strategies-full.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/si-strategies-full.svg + .png')


if __name__ == '__main__':
  main()
