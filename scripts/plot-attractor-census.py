#!/usr/bin/env python3
'''Attractor landscape across the order to chaos transition (SI figure).

Reads the per initial condition and per attractor records written by the
attractor_census binary for a sweep of gamma values at fixed N, and makes
a four panel summary.

Panel a shows the fraction of initial conditions that have reached their
attractor by time t, one curve per gamma. Trajectories that do not recur
within the step budget are right censored, so curves that plateau below
one are lower bounds. The vertical line marks the trajectory length used
in the classification experiments.

Panel b shows cycle lengths, panel c the number of distinct attractors
found per network, and panel d the share of initial conditions absorbed
by the largest basin. The number of attractors is a lower bound censored
by the number of sampled initial conditions, shown as a dashed ceiling.

Usage:
  python scripts/plot-attractor-census.py \
    --data-dir census-data --t-experiment 1000 --gamma-c 2.09 \
    --out-dir plots/si-census
'''
import argparse
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
  'font.size': 16,
  'mathtext.fontset': 'cm',
  'svg.fonttype': 'none',
})

CHAOS_COLOR = '#d62728'
FROZEN_COLOR = '#1f77b4'


def gamma_color(g, gammas, gamma_c):
  '''Diverging color by distance from the critical point.'''
  lo, hi = min(gammas), max(gammas)
  if g < gamma_c:
    x = (gamma_c - g) / max(gamma_c - lo, 1e-9)
    return plt.get_cmap('Reds')(0.35 + 0.6 * x)
  x = (g - gamma_c) / max(hi - gamma_c, 1e-9)
  return plt.get_cmap('Blues')(0.35 + 0.6 * x)


def ci95(values):
  v = np.asarray(values, dtype=float)
  return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data-dir', type=str, required=True)
  p.add_argument('--t-experiment', type=int, default=1000)
  p.add_argument('--gamma-c', type=float, default=2.09)
  p.add_argument('--max-steps', type=int, default=20000)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  data_dir = pathlib.Path(args.data_dir)
  gammas = sorted(float(m.group(1)) for f in data_dir.glob('ics-census-g*.csv')
                  if (m := re.search(r'ics-census-g([\d.]+)\.csv', f.name)))
  ics = {g: pd.read_csv(data_dir / f'ics-census-g{g}.csv') for g in gammas}

  fig, axes2 = plt.subplots(2, 2, figsize=(10.6, 8.6))
  fig.subplots_adjust(hspace=0.36, wspace=0.30)
  axes = axes2.ravel()
  ax_a, ax_b, ax_c, ax_d = axes

  # a: fraction of ICs on their attractor by time t
  tgrid = np.unique(np.round(np.logspace(0, np.log10(args.max_steps), 300)))
  for g in gammas:
    ic = ics[g]
    tr = np.where(ic.converged, ic.transient, np.inf)
    frac = np.searchsorted(np.sort(tr), tgrid, side='right') / len(tr)
    ax_a.plot(tgrid, frac, color=gamma_color(g, gammas, args.gamma_c),
              lw=1.8, label=f'{g:.1f}')
  ax_a.axvline(args.t_experiment, color='#666666', lw=1.0, ls='--')
  ax_a.text(args.t_experiment, 1.045, '$T$', fontsize=15, color='#666666',
            ha='center')
  ax_a.set_xscale('log')
  ax_a.set_xlabel('Time $t$')
  ax_a.set_ylabel('Fraction of ICs on attractor')
  ax_a.set_ylim(0, 1.02)
  ax_a.legend(title='Degree exponent, $\\gamma$', fontsize=11.5, title_fontsize=12,
              loc='lower right', ncol=2, frameon=False, borderaxespad=0.1,
              columnspacing=1.0, handlelength=1.4)

  # b to d: summaries vs gamma with 95 percent CIs over networks
  def per_network(g, fn):
    ic = ics[g]
    return ic.groupby('network_idx').apply(fn, include_groups=False)

  # networks where no IC recurs within the step budget are censored (NaN)
  def guarded(fn):
    return lambda d: fn(d.loc[d.converged], len(d)) if d.converged.any() else np.nan

  med_period = {g: per_network(g, guarded(lambda c, n: c['period'].median()))
                for g in gammas}
  p90_period = {g: per_network(g, guarded(lambda c, n: c['period'].quantile(0.9)))
                for g in gammas}
  n_att = {g: per_network(g, guarded(lambda c, n: c['attractor_key'].nunique()))
           for g in gammas}
  max_basin = {g: per_network(
      g, guarded(lambda c, n: c['attractor_key'].value_counts().iloc[0] / n))
      for g in gammas}
  n_ics = max(len(ics[g]) // ics[g].network_idx.nunique() for g in gammas)

  def quartile_series(stat):
    '''Median across networks with quartile error bars, safe on log axes.'''
    q25, q50, q75 = (np.array([stat[g].quantile(q) for g in gammas])
                     for q in (0.25, 0.5, 0.75))
    return q50, np.vstack([q50 - q25, q75 - q50])

  for ax, stat, label, log in [
      (ax_b, med_period, 'Cycle length', True),
      (ax_c, n_att, 'Attractors sampled per network', True),
      (ax_d, max_basin, 'Largest basin share', False)]:
    m, e = quartile_series(stat)
    ax.errorbar(gammas, m, yerr=e, color='#0f3560', lw=1.8,
                marker='o', markersize=4.5, capsize=2.5)
    if ax is ax_b:
      m90, e90 = quartile_series(p90_period)
      ax.errorbar(gammas, m90, yerr=e90, color='#7a9cc4', lw=1.4,
                  marker='s', markersize=3.5, capsize=2.5)
      ax.text(0.96, 0.86, 'p90', fontsize=15, color='#7a9cc4',
              ha='right', transform=ax.transAxes)
      ax.text(0.96, 0.74, 'median', fontsize=15, color='#0f3560',
              ha='right', transform=ax.transAxes)
    if ax is ax_c:
      ax.axhline(n_ics, color='#999999', lw=1.0, ls='--')
    if log:
      ax.set_yscale('log')
    ax.axvline(args.gamma_c, color='#666666', lw=1.0, ls=':')
    ax.set_xlabel('Degree exponent $\\gamma$')
    ax.set_ylabel(label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  for ax_g in (ax_b, ax_d):
    ax_g.text(args.gamma_c, 1.02, '$\\gamma_c$', fontsize=16,
              color='#666666', ha='center', transform=ax_g.get_xaxis_transform())
  for ax, letter in zip(axes, 'abcd'):
    ax.text(-0.20, 1.04, letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', color='#222222')

  fig.tight_layout()
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'si-census.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'si-census.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/si-census.svg + .png ({len(gammas)} gammas)')


if __name__ == '__main__':
  main()
