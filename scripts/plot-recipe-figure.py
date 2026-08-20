#!/usr/bin/env python3
'''The set level greedy rule against the evolutionary search (main figure).

One panel per noise level with data available. Each point is one network:
the accuracy of the evolutionary search on the horizontal axis, the greedy
rule on the vertical, with the diagonal marking parity. The greedy rule
adds, at each of eight steps, whichever candidate node most raises the
tenth percentile over shock pairs of the panel's single trial
discriminability, computed from the same pilot replicates the search uses.

Usage:
  python scripts/plot-recipe-figure.py \
    --v2-csvs data/panel-design-v2/rho0.5/combined-v2.csv \
    --eps-labels 1 \
    --recipe greedy-nb-p10 \
    --out-dir plots/fig-recipe
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

GA = '#2ca02c'
RECIPE = '#e377c2'   # matches the rule panel of the strategies figure

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
  p.add_argument('--v2-csvs', nargs='+', required=True)
  p.add_argument('--eps-labels', nargs='+', required=True)
  p.add_argument('--recipe', default='greedy-nb-p10')
  p.add_argument('--out-dir', required=True)
  args = p.parse_args()

  n = len(args.v2_csvs)
  fig, axes = plt.subplots(1, n, figsize=(5.0 * n + 0.6, 5.0), squeeze=False)
  axes = axes[0]
  for ax, csv, eps in zip(axes, args.v2_csvs, args.eps_labels):
    d = pd.read_csv(csv)
    ev = d[d.panel == 'evolved'].set_index('original_network_idx').accuracy
    rc = d[d.panel == args.recipe].set_index('original_network_idx').accuracy
    j = ev.index.intersection(rc.index)
    LO, HI = 0.45, 1.02
    ax.fill_between([LO, HI], [LO, HI], HI, color=RECIPE, alpha=0.06,
                    lw=0, zorder=0)
    ax.plot([LO, HI], [LO, HI], color='#bbbbbb', lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(ev[j], rc[j], s=34, color=RECIPE, alpha=0.75, lw=0, zorder=3)
    w = stats.wilcoxon(rc[j], ev[j])
    frac = (rc[j] > ev[j]).mean()
    ax.set_xlabel('Evolutionary search accuracy')
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect('equal')
    ax.set_title(f'$\\varepsilon = {eps}$', fontsize=19)
    # the lower right triangle is empty by construction, put the stats there
    ax.text(0.96, 0.26, f'rule wins {100*frac:.0f}% of networks', fontsize=15,
            transform=ax.transAxes, va='top', ha='right', color=RECIPE)
    ax.text(0.96, 0.18, f'rule mean {rc[j].mean():.2f}', fontsize=15,
            transform=ax.transAxes, va='top', ha='right', color=RECIPE)
    ax.text(0.96, 0.10, f'search mean {ev[j].mean():.2f}', fontsize=15,
            transform=ax.transAxes, va='top', ha='right', color='#777777')
    print(f'eps {eps}: rule {rc[j].mean():.3f} vs GA {ev[j].mean():.3f}, '
          f'wins {100*frac:.0f}%, Wilcoxon p={w.pvalue:.1e}, n={len(j)}')
  axes[0].set_ylabel('Worst pair rule accuracy')
  axes[0].text(0.60, 0.62, 'rule better', fontsize=14, color=RECIPE,
               rotation=45, ha='center', va='bottom', alpha=0.9)
  for ax, letter in zip(axes, 'abc'):
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes,
            fontsize=27, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'fig-recipe.svg', bbox_inches='tight')
  fig.savefig(out / 'fig-recipe.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out}/fig-recipe.svg + .png')


if __name__ == '__main__':
  main()
