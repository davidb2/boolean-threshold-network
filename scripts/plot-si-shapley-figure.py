#!/usr/bin/env python3
'''Why insensitive reporters carry information (SI figure).

Every panel member is scored by its exact Shapley value in the cooperative
game whose value function is the classification accuracy of a sub panel.
The deep ablation campaign evaluated every one of the 254 removal subsets
of each evolved eight member panel, which is the complete coalition
lattice, so the Shapley values are exact rather than sampled. The empty
panel is assigned chance accuracy. The values satisfy the efficiency
axiom by construction: they sum to the accuracy of the intact panel minus
chance, which is checked and printed.

  a  Shapley value against sensitivity for every member of every evolved
     panel at high noise. Insensitive members carry substantial credit.
  b  mean Shapley value by class across every noise level with deep
     ablation data. The credit assigned to sensitive members falls with
     noise while that assigned to insensitive members rises, so the two
     classes converge.
  c  how often an insensitive member outranks the median sensitive member
     of its own panel, against noise.
  d  where in the insensitive range the evolutionary search recruits.
     Selected insensitive members sit at the top of the insensitive range
     rather than at its floor, so the search avoids dead nodes.

Usage:
  python scripts/plot-si-shapley-figure.py \
    --deep-dir data/sensitivity --sensitivity-dir data/sensitivity \
    --sweep-dir data/drug-rho-sweep \
    --ga-csv-99 ... --ga-csv-50 ... --out-dir plots/si-shapley
'''
import argparse
import ast
import pathlib
import re
from math import factorial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENS = '#ff7f0e'
INSENS = '#000000'
CHANCE = 1 / 11
K = 8

plt.rcParams.update({
  'font.size': 19,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def shapley(g):
  '''Exact Shapley values from the full removal lattice of one panel.'''
  panel = sorted({int(ast.literal_eval(r)[0]) for r in g[g.m_removed == 1].removed_nodes})
  if len(panel) != K:
    return None, None
  idx = {p: i for i, p in enumerate(panel)}
  v = {frozenset(range(K)): g.baseline_acc.iloc[0], frozenset(): CHANCE}
  for _, r in g.iterrows():
    rem = [idx[int(x)] for x in ast.literal_eval(r.removed_nodes)]
    v[frozenset(set(range(K)) - set(rem))] = r.ablated_acc
  phi = np.zeros(K)
  for j in range(K):
    others = [i for i in range(K) if i != j]
    for mask in range(1 << (K - 1)):
      S = frozenset(others[b] for b in range(K - 1) if mask >> b & 1)
      k = len(S)
      phi[j] += factorial(k) * factorial(K - k - 1) / factorial(K) * (v[S | {j}] - v[S])
  return panel, phi


def collect(deep_dir, sens_dir, min_baseline=0.7):
  rows, checks = [], []
  for p in sorted(pathlib.Path(deep_dir).glob('ablation-k8-deep-rho*.csv')):
    m = re.match(r'ablation-k8-deep-rho([\d.]+)(?:-(b\d))?\.csv', p.name)
    rho, batch = m.group(1), (m.group(2) or 'b1')
    bpath = pathlib.Path(sens_dir) / f'B-rho{rho}.npz'
    if not bpath.exists():
      print(f'  no B array for rho {rho}, skipping {p.name}')
      continue
    d = pd.read_csv(p)
    if d.groupby('original_network_idx')['baseline_acc'].first().mean() < min_baseline:
      print(f'  skipping {p.name}: baseline below guard')
      continue
    b = np.load(bpath)
    B, bnets = b['B'], [int(x) for x in b['networks']]
    cut = antimode(B)
    eps = 2 * (1 - float(rho))
    for net, g in d.groupby('original_network_idx'):
      panel, phi = shapley(g)
      if panel is None:
        continue
      bi = bnets.index(int(net))
      checks.append(abs(phi.sum() - (g.baseline_acc.iloc[0] - CHANCE)))
      for node, ph in zip(panel, phi):
        rows.append(dict(eps=eps, cohort=f'{rho}-{batch}', net=int(net), node=node,
                         S=float(B[bi, node]), phi=float(ph),
                         sens=bool(B[bi, node] > cut), cut=cut))
  print(f'efficiency axiom: max |sum(phi) - (baseline - chance)| = {max(checks):.2e}')
  return pd.DataFrame(rows)


def panel_recruitment(ga_csv, b_file, rng):
  '''B of selected insensitive members vs all insensitive nodes.'''
  b = np.load(b_file)
  B, bnets = b['B'], [int(x) for x in b['networks']]
  cut = antimode(B)
  ga = pd.read_csv(ga_csv)
  ga = ga[ga.max_num_features == K]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  sel, avail = [], []
  for _, r in fin.iterrows():
    nodes = [int(s.split('-')[1]) for s in ast.literal_eval(r['features'])]
    if len(set(nodes)) != K:
      continue
    bi = bnets.index(int(r['original_network_idx']))
    row = B[bi]
    sel.extend(row[n] for n in nodes if row[n] <= cut)
    ins = np.where(row <= cut)[0]
    avail.extend(row[rng.choice(ins, size=min(200, len(ins)), replace=False)])
  return np.array(sel), np.array(avail), cut


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--deep-dir', type=str, required=True)
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  rng = np.random.default_rng(11)
  R = collect(args.deep_dir, args.sensitivity_dir)
  print(f'{R.cohort.nunique()} cohorts, {len(R)} panel members')

  fig = plt.figure(figsize=(15.4, 9.4))
  gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
  ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
  ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

  # a: Shapley vs sensitivity at high noise
  hi = R[np.isclose(R.eps, 1.0)]
  for lab, sub, col in [('insensitive', hi[~hi.sens], INSENS),
                        ('sensitive', hi[hi.sens], SENS)]:
    ax_a.scatter(sub.S, sub.phi, s=16, color=col, alpha=0.45, lw=0, label=lab)
  ax_a.axvline(hi.cut.iloc[0], color='#e8a000', lw=1.6, linestyle=(0, (4, 3)))
  ax_a.axhline(0, color='#bbbbbb', lw=1.0)
  ax_a.set_xlabel('Sensitivity, $S$')
  ax_a.set_ylabel('Shapley value')
  ax_a.set_title('$\\varepsilon = 1$', fontsize=19)
  ax_a.legend(frameon=False, fontsize=14, loc='upper left', handletextpad=0.2)

  # b: mean Shapley by class across noise
  for lab, sub, col in [('sensitive', R[R.sens], SENS), ('insensitive', R[~R.sens], INSENS)]:
    g = sub.groupby('eps')['phi'].agg(['mean', 'sem'])
    ax_b.fill_between(g.index, g['mean'] - 1.96 * g['sem'], g['mean'] + 1.96 * g['sem'],
                      color=col, alpha=0.18, lw=0)
    ax_b.plot(g.index, g['mean'], color=col, lw=2.2, marker='o', markersize=4.5,
              label=lab)
  ax_b.set_xscale('symlog', linthresh=0.02, linscale=0.55)
  ax_b.set_xticks([0, 0.02, 0.1, 0.5, 1])
  ax_b.set_xticklabels(['0', '0.02', '0.1', '0.5', '1'])
  ax_b.set_xlim(-0.004, 1.35)
  ax_b.set_ylim(0, None)
  ax_b.set_xlabel('Noise, $\\varepsilon$')
  ax_b.set_ylabel('Mean Shapley value')
  ax_b.legend(frameon=False, fontsize=14, loc='center left')

  # c: how often an insensitive member outranks the panel median sensitive member
  def frac_beat(g):
    s, i = g[g.sens], g[~g.sens]
    if not len(s) or not len(i):
      return np.nan
    return float((i.phi > s.phi.median()).mean())
  per = (R.groupby(['eps', 'cohort', 'net']).apply(frac_beat, include_groups=False)
           .rename('frac').reset_index().dropna())
  gg = per.groupby('eps')['frac'].agg(['mean', 'sem'])
  ax_c.fill_between(gg.index, gg['mean'] - 1.96 * gg['sem'], gg['mean'] + 1.96 * gg['sem'],
                    color=INSENS, alpha=0.18, lw=0)
  ax_c.plot(gg.index, gg['mean'], color=INSENS, lw=2.2, marker='o', markersize=4.5)
  ax_c.set_xscale('symlog', linthresh=0.02, linscale=0.55)
  ax_c.set_xticks([0, 0.02, 0.1, 0.5, 1])
  ax_c.set_xticklabels(['0', '0.02', '0.1', '0.5', '1'])
  ax_c.set_xlim(-0.004, 1.35)
  ax_c.set_ylim(0, None)
  ax_c.set_xlabel('Noise, $\\varepsilon$')
  ax_c.set_ylabel('Fraction above the panel\nmedian sensitive member')

  # d: where the search recruits inside the insensitive range
  sel, avail, cut = panel_recruitment(args.ga_csv_50,
                                      f'{args.sensitivity_dir}/B-rho0.5.npz', rng)
  bins = np.linspace(0, cut, 26)
  ax_d.hist(avail, bins=bins, density=True, color='#c7c7c7', lw=0,
            label='all insensitive nodes')
  ax_d.hist(sel, bins=bins, density=True, histtype='step', color=INSENS, lw=2.2,
            label='selected by the search')
  ax_d.axvline(np.median(avail), color='#7f7f7f', lw=1.4, linestyle=(0, (4, 3)))
  ax_d.axvline(np.median(sel), color=INSENS, lw=1.4, linestyle=(0, (4, 3)))
  ax_d.set_xlabel('Sensitivity, $S$ (insensitive range only)')
  ax_d.set_ylabel('Density')
  ax_d.set_title('$\\varepsilon = 1$', fontsize=19)
  ax_d.legend(frameon=False, fontsize=14, loc='upper left')
  print(f'recruitment: median S selected {np.median(sel):.3f} vs available '
        f'{np.median(avail):.3f} (cutoff {cut:.3f})')

  for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], 'abcd'):
    ax.text(-0.19, 1.06, letter, transform=ax.transAxes,
            fontsize=30, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'si-shapley.svg', bbox_inches='tight')
  fig.savefig(out / 'si-shapley.png', bbox_inches='tight', dpi=300)
  R.to_csv(out / 'si-shapley-data.csv', index=False)
  print(f'wrote {out}/si-shapley.svg + .png')


if __name__ == '__main__':
  main()
