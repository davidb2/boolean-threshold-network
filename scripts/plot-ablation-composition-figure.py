#!/usr/bin/env python3
'''Ablation composition figure: what it costs to remove sensitive versus
insensitive reporters, and how noise changes the answer.

  a  the direct view (modeled on the original exploratory panel): accuracy
     after removal versus the number of members removed, each removal
     subset classified as all sensitive, mixed, or all insensitive.
     One column per noise level (eps = 0, 0.5, 1), unit means as jittered
     points, class means with 95 percent CIs on top, baseline dashed.
  b  the paired premium of removing one sensitive instead of one
     insensitive member, across all fifteen noise levels. The premium is
     a step: elevated only near eps = 0, flat from eps = 0.05 to 1.
  c  the accuracy drop from removing one member as a function of the mean
     sensitivity B of that member. Damage is a smooth function of B with
     no jump at the class cutoff, so the dichotomy is a discretization.
  d  the anchoring synergy: the cost of removing one insensitive member,
     for an intact panel and after one sensitive member is already gone.

Statistical unit is one (cohort, network) panel. Values are aggregated
within a unit before any statistics. Cells backed by fewer than 10 units
are not drawn. Composition classes: all sensitive means every removed
node is above the per network sensitivity cutoff, all insensitive means
none is.

Usage:
  python scripts/plot-ablation-composition-figure.py \
    --deep-dir data/sensitivity --out-dir plots/fig-ablation-composition
'''
import argparse
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENS = '#ff7f0e'
INSENS = '#000000'
MIXED = '#7f7f7f'
# darker green = lower noise, as in the panel size and grand figures
EPS_GREEN = {0.0: '#14571a', 0.5: '#2ca02c', 1.0: '#98df8a'}
CHANCE = 1 / 11
MIN_UNITS = 10

plt.rcParams.update({
  'font.size': 20,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def load_all(deep_dir):
  frames = []
  for p in sorted(pathlib.Path(deep_dir).glob('ablation-k8-deep-rho*.csv')):
    m = re.match(r'ablation-k8-deep-rho([\d.]+)(?:-(b\d))?\.csv', p.name)
    rho, batch = float(m.group(1)), (m.group(2) or 'b1')
    d = pd.read_csv(p)
    if d.groupby('original_network_idx')['baseline_acc'].first().mean() < 0.7:
      print(f'  skipping {p.name}: baseline below guard')
      continue
    d['eps'] = 2 * (1 - rho)
    d['unit'] = f'{rho}-{batch}-' + d['original_network_idx'].astype(str)
    d['cohort'] = f'{rho}-{batch}'
    frames.append(d)
  return pd.concat(frames, ignore_index=True)


def classify(d):
  cls = np.where(d.n_sensitive_removed == d.m_removed, 'all sensitive',
                 np.where(d.n_sensitive_removed == 0, 'all insensitive', 'mixed'))
  return d.assign(cls=cls)


def panel_a(axes, df):
  eps_levels = [0.0, 0.5, 1.0]
  order = [('all insensitive', INSENS, -0.28), ('mixed', MIXED, 0.0),
           ('all sensitive', SENS, 0.28)]
  rng = np.random.default_rng(3)
  for ax, eps in zip(axes, eps_levels):
    d = classify(df[np.isclose(df.eps, eps) & (df.m_removed <= 4)])
    base = d.groupby('unit')['baseline_acc'].first().mean()
    ax.axhline(base, color='#555555', lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(CHANCE, color='#bbbbbb', lw=1.0, linestyle=(0, (2, 2)), zorder=1)
    per = (d.groupby(['unit', 'm_removed', 'cls'])['ablated_acc']
             .mean().rename('acc').reset_index())
    for cls, color, off in order:
      g = per[per.cls == cls]
      counts = g.groupby('m_removed')['acc'].count()
      keep = counts[counts >= MIN_UNITS].index
      g = g[g.m_removed.isin(keep)]
      x = g.m_removed + off + rng.uniform(-0.09, 0.09, len(g))
      ax.scatter(x, g.acc, s=7, color=color, alpha=0.25, lw=0, zorder=2)
      mm = g.groupby('m_removed')['acc'].agg(['mean', 'sem'])
      ax.errorbar(mm.index + off, mm['mean'], yerr=1.96 * mm['sem'],
                  fmt='_', color=color, markersize=16, markeredgewidth=3.0,
                  capsize=0, elinewidth=1.6, zorder=4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Members removed')
    ax.set_title(f'$\\varepsilon = {eps:g}$', fontsize=21)
  axes[0].set_ylabel('Classification accuracy')
  axes[0].text(4.5, CHANCE + 0.02, 'chance', fontsize=13, color='#999999', ha='right')
  handles = [plt.Line2D([], [], color=c, marker='_', markersize=13,
                        markeredgewidth=3, lw=0, label=l)
             for l, c, _ in order]
  handles.append(plt.Line2D([], [], color='#555555', lw=1.2,
                            linestyle=(0, (4, 3)), label='no removal'))
  axes[0].legend(handles=handles, frameon=False, fontsize=14,
                 loc='lower left', handlelength=1.1, borderaxespad=0.2,
                 labelspacing=0.3)


def panel_b(ax, df):
  m1 = df[df.m_removed == 1]
  per = (m1.groupby(['cohort', 'eps', 'unit', 'n_sensitive_removed'])['acc_drop']
           .mean().unstack('n_sensitive_removed'))
  per = per.dropna(subset=[0, 1])
  per['gap'] = per[1] - per[0]
  coh = per.groupby(['cohort', 'eps'])['gap'].mean().reset_index()
  plateau = coh[coh.eps >= 0.05]['gap']
  ax.axhspan(plateau.mean() - 2 * plateau.std(), plateau.mean() + 2 * plateau.std(),
             color='#c7c7c7', alpha=0.35, lw=0, zorder=1)
  ax.axhline(plateau.mean(), color='#7f7f7f', lw=1.2, zorder=2)
  ax.scatter(coh.eps, coh.gap, s=52, color='#222222', zorder=4)
  ax.set_xscale('symlog', linthresh=0.02, linscale=0.55)
  ax.set_xticks([0, 0.02, 0.1, 0.5, 1])
  ax.set_xticklabels(['0', '0.02', '0.1', '0.5', '1'])
  ax.set_xlim(-0.004, 1.35)
  ax.set_ylim(0, 0.105)
  ax.set_xlabel('Noise, $\\varepsilon$')
  ax.set_ylabel('Sensitive premium\n(paired, one removal)')
  ax.text(0.96, plateau.mean() + 2 * plateau.std() + 0.004, 'plateau band',
          fontsize=14, color='#7f7f7f', ha='right')


def panel_c(ax, df):
  m1 = df[(df.m_removed == 1) & np.isin(np.round(df.eps, 2), [0.0, 0.5, 1.0])]
  bins = np.arange(0, 0.65, 0.05)
  mids = 0.5 * (bins[:-1] + bins[1:])
  for eps in [0.0, 0.5, 1.0]:
    d = m1[np.isclose(m1.eps, eps)].copy()
    d['bin'] = pd.cut(d.meanB_removed, bins, labels=False)
    per = d.groupby(['unit', 'bin'])['acc_drop'].mean().reset_index()
    g = per.groupby('bin')['acc_drop'].agg(['mean', 'sem', 'count'])
    g = g[g['count'] >= MIN_UNITS]
    x = mids[g.index.astype(int)]
    color = EPS_GREEN[eps]
    ax.fill_between(x, g['mean'] - 1.96 * g['sem'], g['mean'] + 1.96 * g['sem'],
                    color=color, alpha=0.18, lw=0)
    ax.plot(x, g['mean'], color=color, lw=2.2, marker='o', markersize=4.5,
            label=f'$\\varepsilon = {eps:g}$')
  ax.axvspan(0.30, 0.36, color='#e8a000', alpha=0.18, lw=0)
  ax.text(0.33, 0.005, 'class\ncutoff', fontsize=13, color='#a97a00',
          ha='center', va='bottom')
  ax.set_xlabel('Sensitivity $B$ of removed member')
  ax.set_ylabel('Accuracy drop')
  ax.set_xlim(0, 0.62)
  ax.set_ylim(0, 0.19)
  ax.legend(frameon=False, fontsize=15, loc='upper left', handlelength=1.2)


def panel_d(ax, df):
  labels = {0: 'panel intact', 1: 'one sensitive\nalready removed'}
  width = 0.34
  d12 = df[df.m_removed <= 2]
  res = {}
  for eps in [0.0, 0.5, 1.0]:
    d = d12[np.isclose(d12.eps, eps)]
    per = (d.groupby(['unit', 'm_removed', 'n_sensitive_removed'])['acc_drop']
             .mean().unstack(['m_removed', 'n_sensitive_removed']))
    c0 = per.get((1, 0))
    c1 = per.get((2, 1)) - per.get((1, 1))
    both = pd.DataFrame({'c0': c0, 'c1': c1}).dropna()
    res[eps] = both
  xs = np.arange(3, dtype=float)
  for j, s in enumerate([0, 1]):
    vals = [res[e][f'c{s}'].mean() for e in [0.0, 0.5, 1.0]]
    errs = [1.96 * res[e][f'c{s}'].sem() for e in [0.0, 0.5, 1.0]]
    ax.bar(xs + (j - 0.5) * width, vals, width * 0.92, yerr=errs,
           color='#ffffff' if s == 0 else INSENS,
           edgecolor=INSENS, linewidth=1.4,
           error_kw=dict(elinewidth=1.4, capsize=3), label=labels[s])
  ax.set_xticks(xs)
  ax.set_xticklabels(['$\\varepsilon = 0$', '$\\varepsilon = 0.5$', '$\\varepsilon = 1$'])
  ax.set_ylabel('Cost of one\ninsensitive removal')
  ax.set_ylim(0, 0.105)
  ax.legend(frameon=False, fontsize=15, loc='upper left', handlelength=1.1)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--deep-dir', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  df = load_all(args.deep_dir)
  print(f'{df.cohort.nunique()} cohorts, {df.unit.nunique()} units, {len(df)} subsets')

  fig = plt.figure(figsize=(15.6, 10.6))
  gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.6)
  ax_a = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
          fig.add_subplot(gs[0, 4:6])]
  ax_b = fig.add_subplot(gs[1, 0:2])
  ax_c = fig.add_subplot(gs[1, 2:4])
  ax_d = fig.add_subplot(gs[1, 4:6])

  panel_a(ax_a, df)
  panel_b(ax_b, df)
  panel_c(ax_c, df)
  panel_d(ax_d, df)

  for ax, letter in zip([ax_a[0], ax_b, ax_c, ax_d], 'abcd'):
    ax.text(-0.28, 1.05, letter, transform=ax.transAxes,
            fontsize=32, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-ablation-composition.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-ablation-composition.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-ablation-composition.svg + .png')


if __name__ == '__main__':
  main()
