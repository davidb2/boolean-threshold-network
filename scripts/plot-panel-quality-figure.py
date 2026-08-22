#!/usr/bin/env python3
'''Panel quality is a set level property (SI figure).

  a  classification accuracy of the six panel designs at eps = 1, means
     with 95 percent confidence intervals across the 50 networks. The
     margin designed dormant panels beat the sensitivity heuristic and
     the coverage designed ones do not, yet the evolved panels beat both,
     and no design assembled from single node properties reaches them.
  b  what does predict accuracy: the tenth percentile of the covariance
     aware pairwise discriminability, one point per (network, design),
     with the Spearman correlation printed. The same statistic computed
     under the assumption of independent members does not order the
     designs (inset text).

Usage:
  python scripts/plot-panel-quality-figure.py \
    --panels-csv data/panel-design/rho0.5/combined.csv \
    --redundancy-csv data/panel-design/rho0.5/redundancy.csv \
    --out-dir plots/fig-panel-quality
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ORDER = ['evolved', 'greedy-margin', 'half-and-half', 'promiscuous',
         'greedy-cover', 'dormant-random']
LABEL = {'evolved': 'evolved',
         'greedy-margin': 'dormant, designed for margin',
         'half-and-half': 'half promiscuous, half dormant',
         'promiscuous': 'most sensitive',
         'greedy-cover': 'dormant, designed for coverage',
         'dormant-random': 'dormant, random'}
COLOR = {'evolved': '#2ca02c', 'greedy-margin': '#000000',
         'half-and-half': '#7f7f7f', 'promiscuous': '#ff7f0e',
         'greedy-cover': '#555555', 'dormant-random': '#c7c7c7'}

plt.rcParams.update({
  'font.size': 18,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--panels-csv', required=True)
  p.add_argument('--redundancy-csv', required=True)
  p.add_argument('--panel-letters', type=str, default='ab')
  p.add_argument('--out-dir', required=True)
  args = p.parse_args()

  acc = pd.read_csv(args.panels_csv)
  red = pd.read_csv(args.redundancy_csv)

  fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.6, 5.4),
                                   gridspec_kw={'width_ratios': [1.0, 1.15],
                                                'wspace': 0.24})

  # a: accuracy by design
  ys = np.arange(len(ORDER))[::-1]
  for y, k in zip(ys, ORDER):
    s = acc[acc.panel == k].accuracy
    ax_a.errorbar(s.mean(), y, xerr=1.96 * s.sem(), fmt='o', markersize=9,
                  color=COLOR[k], capsize=4, elinewidth=1.8)
  ax_a.set_yticks(ys)
  ax_a.set_yticklabels([LABEL[k] for k in ORDER], fontsize=15)
  ax_a.axvline(1 / 11, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)))
  ax_a.text(1 / 11 + 0.012, ys[-1] - 0.35, 'chance', fontsize=13, color='#999999')
  ax_a.set_xlabel('Classification accuracy')
  ax_a.set_xlim(0, 1.0)
  ax_a.set_title('$\\varepsilon = 1$', fontsize=18)

  # b: the covariance aware low tail against accuracy
  for k in ORDER:
    s = red[red.panel == k]
    ax_b.scatter(s.v_diag, s.accuracy, s=26, color=COLOR[k], alpha=0.65,
                 lw=0, label=LABEL[k])
  shown = red[red.panel.isin(ORDER)]
  rho = stats.spearmanr(shown.v_diag, shown.accuracy)
  ax_b.set_xscale('log')
  ax_b.set_xlabel('Single trial discriminability, pooled variance\n(tenth percentile over shock pairs)')
  ax_b.set_ylabel('Classification accuracy')
  ax_b.text(0.03, 0.97, f'$\\rho = {rho.statistic:.2f}$', fontsize=17,
            transform=ax_b.transAxes, va='top')
  ax_b.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none',
              fontsize=11.5, loc='lower right', handletextpad=0.2,
              labelspacing=0.25, borderaxespad=0.2)
  print(f'spearman v_diag vs acc: {rho.statistic:.3f} (p={rho.pvalue:.1e})')

  for ax, letter, dx in [(ax_a, args.panel_letters[0], -0.62), (ax_b, args.panel_letters[1], -0.16)]:
    ax.text(dx, 1.04, letter, transform=ax.transAxes,
            fontsize=27, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'fig-panel-quality.svg', bbox_inches='tight')
  fig.savefig(out / 'fig-panel-quality.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out}/fig-panel-quality.svg + .png')


if __name__ == '__main__':
  main()
