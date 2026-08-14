#!/usr/bin/env python3
'''Shock-resolved sensitivity disks for one evolved reporter panel.

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

  fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4),
                           subplot_kw=dict(projection='polar'))
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
  cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.03)
  cbar.set_label('Sensitivity to drug, $s_{j,q}$')
  fig.suptitle(
    f'Evolved panel of network {net}, $\\rho = {args.rho}$, wedges are the ten drugs',
    fontsize=12, y=0.99,
  )

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  name = f'fig-disks-rho{args.rho}'
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png (network {net})')


if __name__ == '__main__':
  main()
