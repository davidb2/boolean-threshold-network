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
}
BACKGROUND = ['in-degree', 'out-degree', 'mmse', 'jaccard', 'upstream',
              'entropy-diversity']
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
SPLIT = 6      # answered shocks at or above this = promiscuous


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  sm = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(sm[w])])


def seq_class_matrix(seq_csv, s_file, b_file):
  '''Matrix of pick classes: rows are greedy steps, columns networks.
  0 unresponsive, 1 promiscuous, 2 dormant.'''
  seq = pd.read_csv(seq_csv)
  sd = np.load(s_file, allow_pickle=True)
  S_all = sd['S'].transpose(0, 2, 1)
  snets = [int(x) for x in sd['networks']]
  cut = antimode(np.load(b_file)['B'])
  nets = sorted(seq.original_network_idx.unique())
  n_steps = int(seq.step.max())
  M = np.zeros((n_steps, len(nets)))
  for c, net in enumerate(nets):
    n_row = (S_all[snets.index(int(net))] >= cut).sum(axis=1)
    g = seq[seq.original_network_idx == net].sort_values('step')
    for r, node in enumerate(g.node):
      nj = n_row[int(node)]
      M[r, c] = 0 if nj == 0 else (1 if nj >= SPLIT else 2)
  return M


def draw_seq_grid(ax, M):
  '''Which class the worst pair rule adds at each greedy step, per network.
  Columns are sorted by their number of promiscuous picks, most to least.'''
  from matplotlib.colors import ListedColormap
  T = np.where(M == 1, 0, np.where(M == 2, 1, 2))
  order = sorted(range(M.shape[1]),
                 key=lambda c: (-(M[:, c] == 1).sum(), tuple(T[:, c])))
  M = M[:, order]
  cmap = ListedColormap(['white', '#ff7f0e', '#262626'])
  ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=2, edgecolors='#dddddd',
                linewidth=0.4)
  ax.invert_yaxis()
  ax.set_yticks(np.arange(M.shape[0]) + 0.5)
  ax.set_yticklabels([str(k + 1) for k in range(M.shape[0])], fontsize=13)
  ax.set_xticks([])
  ax.set_ylabel('Greedy step')
  ax.set_xlabel('Networks, sorted by promiscuous picks')
  for side in ['top', 'right', 'left', 'bottom']:
    ax.spines[side].set_visible(False)
  frac = [(M == v).mean() for v in (1, 2, 0)]
  print(f'grid: P {frac[0]:.2f}, D {frac[1]:.2f}, U {frac[2]:.2f} of picks; '
        f'first pick P in {(M[0] == 1).mean()*100:.0f}% of networks')


def draw_seq_bars(ax, M):
  '''Stacked proportions of the classes added at each greedy step, with
  the step on the vertical axis to mirror the grid.'''
  steps = np.arange(1, M.shape[0] + 1)
  fP = (M == 1).mean(axis=1)
  fD = (M == 2).mean(axis=1)
  fU = (M == 0).mean(axis=1)
  ax.barh(steps, fP, color='#ff7f0e', height=0.8)
  ax.barh(steps, fD, left=fP, color='#262626', height=0.8)
  ax.barh(steps, fU, left=fP + fD, color='white', edgecolor='#bbbbbb',
          linewidth=0.8, height=0.8)
  # rows are shared with the grid to the left, so ticks only, no label
  ax.set_ylim(8.5, 0.5)
  ax.set_yticks(steps)
  ax.set_yticklabels([str(k) for k in steps])
  ax.tick_params(labelsize=15, length=0)
  ax.set_xlabel('Proportion of networks', fontsize=16)
  ax.set_xlim(0, 1)
  ax.set_xticks([0, 0.5, 1])
  ax.spines['left'].set_visible(False)


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
  p.add_argument('--rule-curve-csvs', type=str, nargs=3, default=None,
                 help='rule prefix accuracies per noise level, drawn in a to c')
  p.add_argument('--rule-seq-csv', type=str, default=None,
                 help='greedy pick order at the highest noise level, drawn as panel d')
  p.add_argument('--seq-s-file', type=str, default=None)
  p.add_argument('--seq-b-file', type=str, default=None)
  p.add_argument('--rule-name', type=str, default='greedy-mahalanobis')
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  if args.rule_seq_csv:
    fig = plt.figure(figsize=(15.2, 10.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.42], hspace=0.78,
                          wspace=0.14, bottom=0.10)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in axes[1:]:
      ax.sharey(axes[0])
      plt.setp(ax.get_yticklabels(), visible=False)
    ax_d = fig.add_subplot(gs[1, :2])
    ax_e = fig.add_subplot(gs[1, 2])
    # size the grid box to exactly square cells (8 rows x 50 columns) and
    # give the bars the same vertical extent so bar k lines up with row k
    pd_, pe_ = ax_d.get_position(), ax_e.get_position()
    fw, fh = fig.get_size_inches()
    h = pd_.width * fw * (8 / 50) / fh
    ax_d.set_position([pd_.x0, pd_.y1 - h, pd_.width, h])
    ax_e.set_position([pe_.x0 + 0.045, pd_.y1 - h, pe_.width - 0.045, h])
  elif args.rule_v2_csv:
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
  rule_handle = None
  for k, (ax, sdir, gcsv, rdir, eps) in enumerate(zip(axes, args.strategies_dirs,
                                                      args.ga_csvs,
                                                      args.random_dirs,
                                                      args.eps_labels)):
    handles = draw_panel(ax, sdir, gcsv, rdir) or handles
    if args.rule_curve_csvs:
      mean, sem = agg(pd.read_csv(args.rule_curve_csvs[k]))
      x = mean.index.to_numpy()
      ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                      color=RULE_C, alpha=0.16, lw=0, zorder=10)
      rule_handle, = ax.plot(x, mean.to_numpy(), color=RULE_C, lw=2.4,
                             label='worst pair rule', zorder=11,
                             solid_capstyle='round')
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
  if rule_handle is not None:
    handles = handles + [rule_handle]
  if args.rule_seq_csv:
    M = seq_class_matrix(args.rule_seq_csv, args.seq_s_file, args.seq_b_file)
    draw_seq_grid(ax_d, M)
    draw_seq_bars(ax_e, M)
    from matplotlib.patches import Patch
    grid_handles = [Patch(fc='#ff7f0e', label='promiscuous added'),
                    Patch(fc='#262626', label='dormant added'),
                    Patch(fc='white', ec='#bbbbbb', label='unresponsive added')]
    band_top = ax_d.get_position().y1
    band_bot = ax_d.get_position().y0
    fig.legend(handles=handles + [bg_proxy], loc='center', frameon=False,
               fontsize=16, ncol=3, bbox_to_anchor=(0.5, band_top + 0.085),
               columnspacing=1.6)
    fig.legend(handles=grid_handles, loc='center', frameon=False,
               fontsize=16, ncol=3, bbox_to_anchor=(0.5, band_bot - 0.085),
               columnspacing=1.6)
    ax_d.text(-0.105, 1.06, 'd', transform=ax_d.transAxes,
              fontsize=27, fontweight='bold', color='#222222')
    ax_e.text(-0.10, 1.06, 'e', transform=ax_e.transAxes,
              fontsize=27, fontweight='bold', color='#222222')
  elif ax_d is not None:
    draw_rule_panel(ax_d, args.rule_v2_csv, args.rule_name)
    fig.legend(handles=handles + [bg_proxy], loc='center', frameon=False,
               fontsize=18, ncol=2, bbox_to_anchor=(0.67, 0.26),
               columnspacing=1.6, labelspacing=0.7)
    ax_d.text(-0.19, 1.05, 'd', transform=ax_d.transAxes,
              fontsize=27, fontweight='bold', color='#222222')
  else:
    fig.legend(handles=handles + [bg_proxy], loc='lower center', frameon=False,
               fontsize=17, ncol=3, bbox_to_anchor=(0.5, 0.015), columnspacing=1.4)
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
