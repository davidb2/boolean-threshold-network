#!/usr/bin/env python3
'''Shock-resolved sensitivity profiles of evolved reporter panels.

Panel a shows the disks of one example panel. Because shock identities
are network specific, the aggregate panels use rank space: members are
ranked by mean sensitivity within their panel and each member's shock
responses are sorted, so profiles are comparable across networks.
Panel b shows the mean sorted response profile per member rank over all
networks, and panel c the effective number of shocks each member rank
responds to.

Each disk is one panel member. Wedges are the ten drugs, with wedge
radius and color equal to the member's sensitivity to that drug.
Members are ordered by mean sensitivity, so the spectrum runs from
structured shock-specific profiles to near-empty anchor disks.

The example network is chosen deterministically, the network whose
panel composition is closest to the modal mix with the highest
re-evaluated baseline accuracy.

Usage:
  python scripts/plot-disk-figure.py \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --ga-file data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
    --b-file data/sensitivity/B-rho0.5.npz \
    --ablation-file data/sensitivity/ablation-k8-rho0.5.csv \
    --rho 0.5 --out-dir plots/fig-disks
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
  'font.size': 11,
  'mathtext.fontset': 'cm',
  'svg.fonttype': 'none',
})


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def load_panels(path):
  ga = pd.read_csv(path)
  ga = ga[ga.max_num_features == 8]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  return {int(r['original_network_idx']): [int(s.split('-')[1]) for s in eval(r['features'])]
          for _, r in fin.iterrows() if len(set(eval(r['features']))) == 8}


def pick_network(panels, B, bnets, cut, ablation_file):
  abl = pd.read_csv(ablation_file)
  base = abl.groupby('original_network_idx')['baseline_acc'].first()
  n_sens = {net: int((B[bnets.index(net), nodes] > cut).sum())
            for net, nodes in panels.items()}
  modal = pd.Series(n_sens).mode().iloc[0]
  candidates = [net for net, ns in n_sens.items() if ns == modal and net in base.index]
  return max(candidates, key=lambda net: base[net])


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--s-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--ablation-file', type=str, required=True)
  p.add_argument('--rho', type=str, required=True)
  p.add_argument('--network', type=int, default=None)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  s_data = np.load(args.s_file, allow_pickle=True)
  S, snets = s_data['S'], [int(x) for x in s_data['networks']]
  n_drugs = S.shape[1]
  b_data = np.load(args.b_file)
  B, bnets = b_data['B'], [int(x) for x in b_data['networks']]
  cut = antimode(B)
  panels = load_panels(args.ga_file)

  net = args.network if args.network is not None else pick_network(
    panels, B, bnets, cut, args.ablation_file)
  nodes = panels[net]
  bi = bnets.index(net)
  nodes = sorted(nodes, key=lambda n_: -B[bi, n_])
  si = snets.index(net)

  vmax = max(0.6, S[si][:, nodes].max())
  cmap = plt.get_cmap('viridis')
  theta = np.linspace(0, 2 * np.pi, n_drugs, endpoint=False)
  width = 2 * np.pi / n_drugs * 0.92

  fig = plt.figure(figsize=(12.6, 10.2))
  gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.35)
  axes = np.array([[fig.add_subplot(gs[r, c], projection='polar') for c in range(4)]
                   for r in range(2)])
  for ax, node in zip(axes.ravel(), nodes):
    vals = S[si][:, node]
    ax.bar(theta, vals, width=width, bottom=0,
           color=[cmap(v / vmax) for v in vals],
           edgecolor='white', linewidth=0.6)
    ax.set_ylim(0, vmax)
    ax.set_xticks(theta)
    ax.set_xticklabels([f'{d + 1}' for d in range(n_drugs)], fontsize=8.5)
    ax.set_yticks([0.25, 0.5])
    ax.set_yticklabels(['0.25', '0.5'], fontsize=7)
    ax.tick_params(pad=-2)
    ax.grid(alpha=0.35, lw=0.6)
    ax.spines['polar'].set_color('#cccccc')
    is_sens = B[bi, node] > cut
    label = 'sensitive' if is_sens else 'insensitive'
    color = '#b2543f' if is_sens else '#41618c'
    ax.set_title(f'node {node}\n$\\bar{{B}} = {B[bi, node]:.2f}$, {label}',
                 fontsize=10, color=color, pad=13)

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
  cbar = fig.colorbar(sm, ax=list(axes.ravel()), shrink=0.75, pad=0.02)
  cbar.set_label('Sensitivity to shock, $s_{j,q}$')
  axes[0, 0].text(-0.25, 1.32, 'a', transform=axes[0, 0].transAxes,
                  fontsize=17, fontweight='bold', color='#222222')
  fig.text(0.38, 0.985, f'example panel, network {net}, wedges are the ten shocks',
           fontsize=12, ha='center')

  # aggregate over all networks in rank space
  ranked = []
  eff = []
  for net_j, nodes_j in panels.items():
    if net_j not in snets:
      continue
    bj = bnets.index(net_j)
    sj = snets.index(net_j)
    members = sorted(nodes_j, key=lambda n_: -B[bj, n_])
    prof = np.sort(S[sj][:, members], axis=0)[::-1]   # [shock_rank, member_rank]
    ranked.append(prof.T)                             # [member_rank, shock_rank]
    tot = S[sj][:, members].sum(axis=0)
    sq = (S[sj][:, members] ** 2).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
      e = np.where(sq > 0, tot ** 2 / sq, 0.0)
    eff.append(e)
  ranked = np.array(ranked)   # [nets, 8, 10]
  eff = np.array(eff)         # [nets, 8]

  ax_h = fig.add_subplot(gs[2, 0:2])
  im = ax_h.imshow(ranked.mean(axis=0), aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
  ax_h.set_xticks(range(10))
  ax_h.set_xticklabels([str(i + 1) for i in range(10)], fontsize=9)
  ax_h.set_yticks(range(8))
  ax_h.set_yticklabels([str(i + 1) for i in range(8)], fontsize=9)
  ax_h.set_xlabel('Shock rank within member')
  ax_h.set_ylabel('Member rank within panel')
  ax_h.text(-0.16, 1.13, 'b', transform=ax_h.transAxes,
            fontsize=17, fontweight='bold', color='#222222')
  ax_h.set_title(f'mean sorted profile, {ranked.shape[0]} networks', fontsize=11)

  ax_e = fig.add_subplot(gs[2, 2:4])
  m = eff.mean(axis=0)
  s_ = 1.96 * eff.std(axis=0) / np.sqrt(eff.shape[0])
  ax_e.errorbar(range(1, 9), m, yerr=s_, color='#0f3560', lw=2.0,
                marker='o', markersize=5, capsize=3)
  ax_e.set_xlabel('Member rank within panel')
  ax_e.set_ylabel('Effective number of\nshocks responded to')
  ax_e.set_ylim(0, 10)
  ax_e.set_xticks(range(1, 9))
  ax_e.spines['top'].set_visible(False)
  ax_e.spines['right'].set_visible(False)
  ax_e.text(-0.16, 1.13, 'c', transform=ax_e.transAxes,
            fontsize=17, fontweight='bold', color='#222222')
  ax_e.set_title('breadth of response by member rank', fontsize=11)

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  name = f'fig-disks-rho{args.rho}'
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png (network {net})')


if __name__ == '__main__':
  main()
