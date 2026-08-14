#!/usr/bin/env python3
'''Connectivity of evolved reporter panels: four-panel figure.

  a  pairwise distance between panel members vs size-matched random panels
  b  degree distributions of panel members vs the background
  c  mean panel-to-drug distance: min over members of the directed
     distance from each drug's targets, averaged over drugs, for evolved
     panels vs random panels vs sensitivity-matched random panels
  d  distance of the worst-covered drug, same three groups

Generic topology is a null result (a, b): evolved panels are
indistinguishable from random. But panels are specifically aligned to
the shocks (c, d): they sit closer downstream of every drug's targets,
and sensitivity-matched random panels do NOT reproduce this, so the
positioning is not explained by sensitivity composition.

Usage:
  python scripts/plot-connectivity-figure.py \
    --panel-topology data/sensitivity/panel-topology.csv \
    --connectivity data/sensitivity/connectivity-arrays.npz \
    --sensitivity-dir data/sensitivity \
    --ga rho0.5=... rho0.9=... rho0.99=... \
    --out-dir plots/fig-connectivity
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RHO_COLORS = {'0.5': '#86b6ef', '0.9': '#3987e5', '0.99': '#0f3560'}
GRAY = '#8b93a1'
R_MAX = 10

plt.rcParams.update({
  'font.size': 12,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def load_panels(path):
  ga = pd.read_csv(path)
  ga = ga[ga.max_num_features == 8]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  return {int(r['original_network_idx']): [int(s.split('-')[1]) for s in eval(r['features'])]
          for _, r in fin.iterrows() if len(set(eval(r['features']))) == 8}


def ecdf(ax, values, **kw):
  x = np.sort(values[~np.isnan(values)])
  ax.plot(x, np.arange(1, len(x) + 1) / len(x), drawstyle='steps-post', **kw)


def panel_a(ax, topo):
  order = [('random', 'random', GRAY)] + [
    (f'rho{r}', f'$\\rho = {r}$', RHO_COLORS[r]) for r in ['0.5', '0.9', '0.99']]
  rng = np.random.default_rng(3)
  for i, (key, label, color) in enumerate(order):
    if key == 'random':
      vals = topo[topo.rho == 'random'].groupby('network')['mean_pair_dist'].mean().to_numpy()
    else:
      vals = topo[topo.rho == key]['mean_pair_dist'].to_numpy()
    x = i + rng.uniform(-0.14, 0.14, len(vals))
    ax.plot(x, vals, 'o', color=color, markersize=3.6, alpha=0.55, markeredgewidth=0)
    ax.hlines(np.mean(vals), i - 0.24, i + 0.24, color=color, lw=2.4, zorder=5)
  ax.set_xticks(range(4))
  ax.set_xticklabels(['random', '$\\rho{=}0.5$', '$\\rho{=}0.9$', '$\\rho{=}0.99$'])
  ax.set_ylabel('Mean distance\nbetween panel members')
  ax.set_ylim(1.85, 2.62)


def panel_b(ax, topo):
  rs = np.arange(1, R_MAX + 1)
  cols = [f'up_cov_r{r}' for r in rs]
  rnd = topo[topo.rho == 'random'][cols].to_numpy()
  lo, hi = np.percentile(rnd, [2.5, 97.5], axis=0)
  ax.fill_between(rs, lo, hi, color=GRAY, alpha=0.30, lw=0,
                  label='random panels (95%)')
  for r in ['0.5', '0.9', '0.99']:
    m = topo[topo.rho == f'rho{r}'][cols].mean(axis=0)
    ax.plot(rs, m, color=RHO_COLORS[r], lw=1.8, label=f'$\\rho = {r}$')
  ax.set_xlabel('Hops upstream to nearest member, $r$')
  ax.set_ylabel('Fraction of network\nwithin $r$ of the panel')
  ax.set_xlim(1, R_MAX)
  ax.set_ylim(0, 1.02)
  ax.legend(frameon=False, fontsize=9.5, loc='lower right')


def panel_c(ax, conn, ga_panels):
  nets = list(conn['networks'])
  for deg, ls, lab in [(conn['in_deg'], '-', 'in-degree'), (conn['out_deg'], (0, (4, 2)), 'out-degree')]:
    members = []
    for rho, panels in ga_panels.items():
      for net, nodes in panels.items():
        members += list(deg[nets.index(net), nodes])
    ecdf(ax, np.array(members, dtype=float) + 1, color='#1c5cab', lw=1.9,
         linestyle=ls, label=f'members, {lab}')
    ecdf(ax, deg.ravel().astype(float) + 1, color=GRAY, lw=1.5, linestyle=ls,
         label=f'all nodes, {lab}')
  ax.set_xscale('log')
  ax.set_xlabel('Degree $+ 1$')
  ax.set_ylabel('Cumulative fraction')
  ax.set_ylim(0, 1.02)
  ax.legend(frameon=False, fontsize=9.5, loc='lower right')


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def drug_coverage(conn, ga_panels, sens_dir, n_rand=200, seed=21):
  nets = list(conn['networks'])
  dist = conn['dist_from_targets'].astype(float)
  dist[dist < 0] = np.inf
  rng = np.random.default_rng(seed)
  rows = []
  for rho, panels in ga_panels.items():
    b = np.load(f'{sens_dir}/B-rho{rho}.npz')
    B, bnets = b['B'], [int(x) for x in b['networks']]
    cut = antimode(B)
    for net, nodes in panels.items():
      i = nets.index(net)
      bi = bnets.index(net)
      per_drug = dist[i][:, nodes].min(axis=1)
      rows.append(dict(rho=rho, network=net, group='evolved',
                       mean_d=per_drug.mean(), worst_d=per_drug.max()))
      brow = B[bi]
      sens_pool = np.flatnonzero(brow > cut)
      insens_pool = np.flatnonzero(brow <= cut)
      n_s = int((brow[nodes] > cut).sum())
      acc = {'random': [], 'matched': []}
      for _ in range(n_rand):
        rn = rng.choice(5000, 8, replace=False)
        pd_r = dist[i][:, rn].min(axis=1)
        acc['random'].append((pd_r.mean(), pd_r.max()))
        mn = np.concatenate([rng.choice(sens_pool, n_s, replace=False),
                             rng.choice(insens_pool, 8 - n_s, replace=False)])
        pd_m = dist[i][:, mn].min(axis=1)
        acc['matched'].append((pd_m.mean(), pd_m.max()))
      for group, vals in acc.items():
        v = np.array(vals)
        rows.append(dict(rho=rho, network=net, group=group,
                         mean_d=v[:, 0].mean(), worst_d=v[:, 1].mean()))
  return pd.DataFrame(rows)


GROUPS = [('evolved', '#0f3560', 'evolved'),
          ('matched', '#eb6834', 'random, sensitivity matched'),
          ('random', GRAY, 'random')]


def panel_cd(ax, cov, col, ylabel):
  rhos = ['0.5', '0.9', '0.99']
  width = 0.24
  for gi, (group, color, label) in enumerate(GROUPS):
    xs, ms, ss = [], [], []
    for xi, rho in enumerate(rhos):
      vals = cov[(cov.group == group) & (cov.rho == rho)][col]
      xs.append(xi + (gi - 1) * width)
      ms.append(vals.mean())
      ss.append(1.96 * vals.sem())
    ax.bar(xs, ms, width=width * 0.88, color=color, label=label, zorder=3)
    ax.errorbar(xs, ms, yerr=ss, fmt='none', ecolor='#333333', lw=1.1, capsize=2.5, zorder=4)
  ax.set_xticks(range(len(rhos)))
  ax.set_xticklabels([f'$\\rho = {r}$' for r in rhos])
  ax.set_ylabel(ylabel)


def panel_c_shock(ax, cov):
  panel_cd(ax, cov, 'mean_d', 'Panel-to-shock distance\n(mean over shocks, hops)')
  ax.set_ylim(0, 1.9)
  ax.legend(frameon=False, fontsize=9, loc='upper center', ncol=1)


def panel_d_shock(ax, cov):
  panel_cd(ax, cov, 'worst_d', 'Worst-covered shock\ndistance (hops)')
  ax.set_ylim(0, 2.6)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--panel-topology', type=str, required=True)
  p.add_argument('--connectivity', type=str, required=True)
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--ga', type=str, nargs='+', required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  topo = pd.read_csv(args.panel_topology)
  conn = np.load(args.connectivity)
  ga_panels = {}
  for spec in args.ga:
    label, path = spec.split('=', 1)
    ga_panels[label.replace('rho', '')] = load_panels(path)

  cov = drug_coverage(conn, ga_panels, args.sensitivity_dir)
  fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
  fig.subplots_adjust(hspace=0.38, wspace=0.32)
  panel_a(axes[0, 0], topo)
  panel_c(axes[0, 1], conn, ga_panels)
  panel_c_shock(axes[1, 0], cov)
  panel_d_shock(axes[1, 1], cov)
  for ax, letter in zip(axes.ravel(), 'abcd'):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-connectivity.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-connectivity.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/fig-connectivity.svg + .png')


if __name__ == '__main__':
  main()
