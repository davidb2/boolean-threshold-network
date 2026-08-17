#!/usr/bin/env python3
'''Why the sensitivity distribution is bimodal (SI figure).

  a  one representative network: the distribution of node sensitivity S on a
     linear count scale, split into nodes that are frozen in the control
     dynamics and nodes that are active. The low mode is the frozen
     population and the high mode is the active one.
  b  every network has its own dip, and the dip sits at a different place
     in each: per network antimode location against the relative depth of
     the dip. Pooling networks therefore blurs the gap.
  c  where the two modes come from. The stored control activity is the
     variance v = p(1-p) of a node's control state, so 2v is the mean
     absolute difference expected if the shocked trajectory decorrelates
     from control while keeping the same activity. Active nodes sit on
     that line: they are desynchronizers, and 2v saturates at 0.5, which
     is where the high mode lands. Nodes above the line have shifted their
     mean state, not merely decorrelated, and frozen nodes (v = 0) can
     only respond that way.
  d  aligning each network on its own antimode before pooling recovers the
     gap that the raw pooled histogram hides.

The relative dip depth is 1 - h_antimode / min(h_left, h_right) on a
smoothed histogram, so 0 means no dip and 1 means the antimode bin is
empty.

Usage:
  python scripts/plot-si-bimodality-figure.py \
    --sensitivity-dir data/sensitivity --rho 0.5 --out-dir plots/si-bimodality
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FROZEN_C = '#1f77b4'
ACTIVE_C = '#d62728'
FROZEN_CUT = 0.02

plt.rcParams.update({
  'font.size': 19,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def dip(x, bins=60, lo=0.02, hi=0.45):
  '''Antimode location and relative depth on a smoothed histogram.'''
  c, e = np.histogram(x, bins=np.linspace(0, max(x.max(), 1e-9), bins))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  if w.sum() < 3:
    return np.nan, np.nan
  i = np.argmin(s[w])
  anti, h = ce[w][i], s[w][i]
  left = s[ce <= anti].max() if (ce <= anti).any() else 0
  right = s[ce > anti].max() if (ce > anti).any() else 0
  denom = min(left, right)
  return anti, (1 - h / denom if denom > 0 else np.nan)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--rho', type=str, default='0.5')
  p.add_argument('--example-network', type=int, default=None)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  d = pathlib.Path(args.sensitivity_dir)
  S = np.load(d / f'B-rho{args.rho}.npz')['B']
  A = np.load(d / f'activity-rho{args.rho}.npz')['activity']
  eps = 2 * (1 - float(args.rho))
  n_net = S.shape[0]

  antis, depths = np.array([dip(S[n]) for n in range(n_net)]).T
  ok = ~np.isnan(depths)

  # representative network: median dip depth among those with a clear dip
  if args.example_network is None:
    cand = np.where(ok & (depths > 0.2))[0]
    ex = int(cand[np.argsort(depths[cand])[len(cand) // 2]])
  else:
    ex = args.example_network

  fig = plt.figure(figsize=(15.4, 9.4))
  gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  ax_c = fig.add_subplot(gs[1, 0])
  ax_d = fig.add_subplot(gs[1, 1])

  # a: one network, linear counts, split by control activity
  s_ex, a_ex = S[ex], A[ex]
  bins = np.linspace(0, max(s_ex.max(), 0.6), 61)
  froz, actv = s_ex[a_ex < FROZEN_CUT], s_ex[a_ex >= FROZEN_CUT]
  ax_a.hist([froz, actv], bins=bins, stacked=True, color=[FROZEN_C, ACTIVE_C],
            label=[f'frozen in control ($a < {FROZEN_CUT}$)', 'active in control'],
            lw=0)
  anti_ex = antis[ex]
  ax_a.axvline(anti_ex, color='#e8a000', lw=2.0, linestyle=(0, (4, 3)))
  ax_a.set_ylim(0, np.histogram(actv, bins=bins)[0].max() * 4.2)
  ax_a.text(anti_ex, ax_a.get_ylim()[1] * 0.42, ' antimode', fontsize=15,
            color='#a97a00')
  ax_a.set_xlabel('Sensitivity, $S$')
  ax_a.set_ylabel('Nodes')
  ax_a.set_title(f'One network ($\\varepsilon = {eps:g}$)', fontsize=19)
  ax_a.legend(frameon=False, fontsize=13.5, loc='upper right', borderaxespad=0.2)

  # b: antimode location vs dip depth, all networks
  ax_b.scatter(antis[ok], depths[ok], s=44, color='#222222', alpha=0.75, lw=0)
  ax_b.axhline(0.2, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)))
  ax_b.text(0.44, 0.23, 'visible dip', fontsize=14, color='#999999', ha='right')
  ax_b.set_xlabel('Antimode location')
  ax_b.set_ylabel('Relative dip depth')
  ax_b.set_ylim(0, 1.02)
  ax_b.set_title(f'{int((depths[ok] > 0.2).sum())} of {n_net} networks', fontsize=19)

  # c: sensitivity against the pure decorrelation reference 2v (v = control variance)
  v_all, s_all = A.ravel(), S.ravel()
  lim = 0.62
  ax_c.hist2d(2 * v_all, s_all, bins=[np.linspace(0, lim, 61), np.linspace(0, lim, 61)],
              cmap='Greys', norm=matplotlib.colors.LogNorm())
  ax_c.plot([0, lim], [0, lim], color=ACTIVE_C, lw=2.0, linestyle=(0, (4, 3)))
  act_m = v_all >= FROZEN_CUT
  r = np.corrcoef(s_all[act_m], 2 * v_all[act_m])[0, 1]
  ax_c.text(0.30, 0.335, 'decorrelation', fontsize=14, color=ACTIVE_C, rotation=32)
  ax_c.annotate('frozen nodes:\nmean state shifted', xy=(0.012, 0.42),
                xytext=(0.13, 0.53), fontsize=13.5, color='#222222',
                arrowprops=dict(arrowstyle='->', color='#222222', lw=1.2))
  ax_c.text(0.335, 0.075, f'active nodes\n$r = {r:.2f}$', fontsize=14, color='#222222',
            ha='center')
  ax_c.set_xlabel('Pure decorrelation, $2v$')
  ax_c.set_ylabel('Sensitivity, $S$')
  ax_c.set_xlim(0, lim)
  ax_c.set_ylim(0, lim)
  ax_c.set_title('All nodes', fontsize=19)

  # d: raw pooled vs pooled after aligning each network on its antimode
  raw = S.ravel()
  aligned = np.concatenate([S[n] - antis[n] for n in range(n_net) if ok[n]])
  ax_d.hist(raw, bins=np.linspace(0, 0.7, 71), color='#bbbbbb', lw=0,
            label='pooled directly')
  ax_d.hist(aligned + np.median(antis[ok]), bins=np.linspace(0, 0.7, 71),
            histtype='step', color='#222222', lw=2.0,
            label='pooled after aligning\neach network')
  ax_d.set_xlabel('Sensitivity, $S$')
  ax_d.set_ylabel('Nodes')
  ax_d.set_ylim(0, np.histogram(raw, bins=np.linspace(0, 0.7, 71))[0][3:].max() * 1.6)
  ax_d.legend(frameon=False, fontsize=14, loc='upper right')

  for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], 'abcd'):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=30, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'si-bimodality.svg', bbox_inches='tight')
  fig.savefig(out / 'si-bimodality.png', bbox_inches='tight', dpi=300)
  print(f'example network {ex}, antimode {antis[ex]:.3f}, depth {depths[ex]:.2f}')
  print(f'dip depth median {np.median(depths[ok]):.2f}, '
        f'{int((depths[ok] > 0.2).sum())}/{n_net} above 0.2')
  print(f'wrote {out}/si-bimodality.svg + .png')


if __name__ == '__main__':
  main()
