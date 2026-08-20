#!/usr/bin/env python3
'''Selection strategy comparison: accuracy vs number of reporters.

Three panels (noise eps = 0, 0.5, 1), plus an optional fourth panel with
the set level greedy rule against the evolutionary search on held out
trials (pass --rule-v2-csv). The curve panels use the search's own
evaluator, where its score is an optimum and therefore biased upward, so
the rule comparison is drawn only from the held out protocol. Foreground curves: genetic
algorithm, random selection, highest sensitivity, greedy information gain
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
  'sensitivity':     {'color': '#ff7f0e', 'label': 'highest sensitivity', 'lw': 2.4, 'zorder': 9},
  'infomax':         {'color': '#17becf', 'label': 'greedy information gain', 'lw': 2.4, 'zorder': 10},
  'anchor-reporter': {'color': '#9467bd', 'label': 'anchors then information gain', 'lw': 2.4, 'zorder': 8},
  'influence':       {'color': '#8c564b', 'label': 'influence maximization', 'lw': 2.4, 'zorder': 6},
}
BACKGROUND = ['in-degree', 'out-degree', 'mmse', 'jaccard', 'upstream',
              'entropy-diversity', 'anchor-sensitivity']
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
  if 'accuracy' in ga.columns:
    # a rescored file: final panels re-evaluated on fresh splits
    return ga[['original_network_idx', 'max_num_features', 'accuracy']]
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


RULE_C = '#e377c2'


def draw_rule_panel(ax, v2_csv, recipe):
  '''The set level rule against the search, held out trials, one point
  per network. This protocol differs from the curve panels, where the
  search's score is the optimum of the evaluator being plotted.'''
  from scipy import stats as sstats
  d = pd.read_csv(v2_csv)
  ev = d[d.panel == 'evolved'].set_index('original_network_idx').accuracy
  rc = d[d.panel == recipe].set_index('original_network_idx').accuracy
  j = ev.index.intersection(rc.index)
  LO, HI = 0.45, 1.02
  ax.fill_between([LO, HI], [LO, HI], HI, color=RULE_C, alpha=0.07, lw=0, zorder=0)
  ax.plot([LO, HI], [LO, HI], color='#bbbbbb', lw=1.2, linestyle=(0, (4, 3)), zorder=1)
  ax.scatter(ev[j], rc[j], s=34, color=RULE_C, alpha=0.8, lw=0, zorder=3)
  w = sstats.wilcoxon(rc[j], ev[j])
  frac = (rc[j] > ev[j]).mean()
  ax.set_xlim(LO, HI)
  ax.set_ylim(LO, HI)
  ax.set_aspect('equal')
  ax.set_xlabel('Evolutionary search accuracy')
  ax.set_ylabel('Set level rule accuracy')
  ax.set_title('$\\varepsilon = 1$, $m = 8$, fresh trials', fontsize=19)
  ax.text(0.60, 0.62, 'rule better', fontsize=14, color=RULE_C,
          rotation=45, ha='center', va='bottom')
  ax.text(0.96, 0.26, f'rule wins {100*frac:.0f}% of networks', fontsize=15,
          transform=ax.transAxes, va='top', ha='right', color=RULE_C)
  ax.text(0.96, 0.18, f'rule mean {rc[j].mean():.2f}', fontsize=15,
          transform=ax.transAxes, va='top', ha='right', color=RULE_C)
  ax.text(0.96, 0.10, f'search mean {ev[j].mean():.2f}', fontsize=15,
          transform=ax.transAxes, va='top', ha='right', color='#777777')
  print(f'rule panel: wins {100*frac:.0f}%, {rc[j].mean():.3f} vs '
        f'{ev[j].mean():.3f}, Wilcoxon p={w.pvalue:.1e}, n={len(j)}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--strategies-dirs', type=str, nargs=3, required=True)
  p.add_argument('--ga-csvs', type=str, nargs=3, required=True)
  p.add_argument('--random-dirs', type=str, nargs=3, required=True)
  p.add_argument('--eps-labels', type=str, nargs=3, required=True)
  p.add_argument('--rule-v2-csv', type=str, default=None)
  p.add_argument('--rule-name', type=str, default='greedy-mahalanobis')
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  if args.rule_v2_csv:
    fig = plt.figure(figsize=(15.2, 11.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.14)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in axes[1:]:
      ax.sharey(axes[0])
      plt.setp(ax.get_yticklabels(), visible=False)
    ax_d = fig.add_subplot(gs[1, 0])
  else:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.9), sharey=True)
    fig.subplots_adjust(wspace=0.14, bottom=0.32)
    ax_d = None
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
  axes[0].text(60, CHANCE + 0.02, 'chance', fontsize=14, color='#999999')

  bg_proxy = plt.Line2D([], [], color=BG_COLOR, lw=1.3, label='other heuristics')
  if ax_d is not None:
    draw_rule_panel(ax_d, args.rule_v2_csv, args.rule_name)
    fig.legend(handles=handles + [bg_proxy], loc='center', frameon=False,
               fontsize=18, ncol=2, bbox_to_anchor=(0.67, 0.26),
               columnspacing=1.6, labelspacing=0.7)
    ax_d.text(-0.19, 1.05, 'd', transform=ax_d.transAxes,
              fontsize=27, fontweight='bold', color='#222222')
  else:
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
