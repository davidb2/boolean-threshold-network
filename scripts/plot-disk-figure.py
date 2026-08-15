#!/usr/bin/env python3
'''Shock profiles of evolved reporter panels as clean disks.

Panel a shows the eight members of one evolved panel as disks with no
axes. Each wedge is one of the ten shocks, with wedge length and color
equal to that member's sensitivity to that shock. Disks are grouped by
class: sensitive members bundled on the left, insensitive members on
the right, each group ordered by mean sensitivity.

Panel b shows the effective number of shocks each member rank responds
to, aggregated across all networks (members ranked by mean sensitivity
within their panel).

With --candidates N the script instead renders a contact sheet of the
N best candidate example networks (highest re-evaluated baseline
accuracy, grouped by panel composition) so the example can be chosen.

Usage:
  python scripts/plot-disk-figure.py \
    --s-file data/sensitivity/S-perdrug-rho0.5.npz \
    --ga-file data/drug-fixed-targets-v7/N5000/ga-results-v7/combined-full.csv \
    --b-file data/sensitivity/B-rho0.5.npz \
    --ablation-file data/sensitivity/ablation-k8-rho0.5.csv \
    --eps-label 1 --out-dir plots/fig-disks [--network 7 | --candidates 4]
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

SENS = '#b2543f'
INSENS = '#41618c'


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


def draw_disk(ax, vals, vmax, cmap, ring_color):
  n = len(vals)
  theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
  width = 2 * np.pi / n * 0.9
  ax.bar(theta, vals, width=width, bottom=0,
         color=[cmap(v / vmax) for v in vals],
         edgecolor='white', linewidth=0.5)
  # faint full circle so empty disks are still visible
  ax.bar(theta, [vmax] * n, width=width, bottom=0, fill=False,
         edgecolor='none', linewidth=0)
  ax.set_ylim(0, vmax)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.grid(False)
  ax.spines['polar'].set_color(ring_color)
  ax.spines['polar'].set_linewidth(1.6)


def panel_groups(nodes, B_row, cut):
  sens = sorted([n for n in nodes if B_row[n] > cut], key=lambda n: -B_row[n])
  insens = sorted([n for n in nodes if B_row[n] <= cut], key=lambda n: -B_row[n])
  return sens, insens


def draw_panel_row(fig, gs_row, S_net, nodes, B_row, cut, vmax, cmap, label=None):
  '''One example panel as a row of grouped, axis free disks.'''
  sens, insens = panel_groups(nodes, B_row, cut)
  order = sens + insens
  n_s = len(sens)
  for k, node in enumerate(order):
    # small horizontal gap between the two groups
    slot = k if k < n_s else k + 1
    ax = fig.add_subplot(gs_row[0, slot], projection='polar')
    draw_disk(ax, S_net[:, node], vmax, cmap, SENS if k < n_s else INSENS)
  if label is not None:
    ax0 = fig.add_subplot(gs_row[0, :], frameon=False)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center', labelpad=34)
  return n_s, len(insens)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--s-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--ablation-file', type=str, required=True)
  p.add_argument('--eps-label', type=str, required=True)
  p.add_argument('--network', type=int, default=None)
  p.add_argument('--candidates', type=int, default=0)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  s_data = np.load(args.s_file, allow_pickle=True)
  S, snets = s_data['S'], [int(x) for x in s_data['networks']]
  b_data = np.load(args.b_file)
  B, bnets = b_data['B'], [int(x) for x in b_data['networks']]
  cut = antimode(B)
  panels = load_panels(args.ga_file)
  abl = pd.read_csv(args.ablation_file)
  base = abl.groupby('original_network_idx')['baseline_acc'].first()

  vmax = 0.6
  cmap = plt.get_cmap('viridis')
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.candidates:
    # top candidates by baseline accuracy within each composition group
    rows = []
    for net, nodes in panels.items():
      if net not in snets or net not in base.index:
        continue
      n_s = int((B[bnets.index(net), nodes] > cut).sum())
      rows.append((net, n_s, float(base[net])))
    df = pd.DataFrame(rows, columns=['net', 'n_sens', 'baseline'])
    modal = int(df.n_sens.mode().iloc[0])
    df['dist'] = (df.n_sens - modal).abs()
    picks = (df.sort_values(['dist', 'baseline'], ascending=[True, False])
               .groupby('n_sens').head(2)
               .sort_values(['dist', 'baseline'], ascending=[True, False])
               .head(args.candidates))
    fig = plt.figure(figsize=(11.5, 1.55 * len(picks)))
    outer = fig.add_gridspec(len(picks), 1, hspace=0.5)
    for i, r in enumerate(picks.itertuples()):
      gs_row = outer[i].subgridspec(1, 9, wspace=0.12)
      si, bi = snets.index(r.net), bnets.index(r.net)
      draw_panel_row(fig, gs_row, S[si], panels[r.net], B[bi], cut, vmax, cmap,
                     label=f'network {r.net}\n{r.n_sens} sens, acc {r.baseline:.2f}')
    fig.savefig(out_dir / 'fig-disks-candidates.png', bbox_inches='tight', dpi=200)
    print(f'wrote {out_dir}/fig-disks-candidates.png ({len(picks)} candidates)')
    return

  net = args.network
  if net is None:
    valid = [n for n in panels if n in snets and n in base.index]
    net = max(valid, key=lambda n: base[n])
  nodes = panels[net]
  si, bi = snets.index(net), bnets.index(net)

  fig = plt.figure(figsize=(12.8, 4.4))
  outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.45)
  gs_row = outer[0].subgridspec(1, 9, wspace=0.12)
  n_s, n_i = draw_panel_row(fig, gs_row, S[si], nodes, B[bi], cut, vmax, cmap)
  fig.text(0.06, 0.965, 'a', fontsize=17, fontweight='bold', color='#222222')
  # group labels centered under each group, computed from the actual axes
  fig.canvas.draw()
  pos = [ax.get_position() for ax in fig.axes if ax.name == 'polar']
  sens_axes, insens_axes = pos[:n_s], pos[n_s:n_s + n_i]
  if sens_axes:
    xc = 0.5 * (sens_axes[0].x0 + sens_axes[-1].x1)
    fig.text(xc, 0.53, 'sensitive members', fontsize=10.5, color=SENS, ha='center')
  if insens_axes:
    xc = 0.5 * (insens_axes[0].x0 + insens_axes[-1].x1)
    fig.text(xc, 0.53, 'insensitive members', fontsize=10.5, color=INSENS, ha='center')

  # panel b: effective number of shocks by member rank, all networks
  eff = []
  for net_j, nodes_j in panels.items():
    if net_j not in snets:
      continue
    bj, sj = bnets.index(net_j), snets.index(net_j)
    members = sorted(nodes_j, key=lambda n_: -B[bj, n_])
    tot = S[sj][:, members].sum(axis=0)
    sq = (S[sj][:, members] ** 2).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
      eff.append(np.where(sq > 0, tot ** 2 / sq, 0.0))
  eff = np.array(eff)
  gs_b = outer[1].subgridspec(1, 3)
  ax_e = fig.add_subplot(gs_b[0, 1])
  m = eff.mean(axis=0)
  se = 1.96 * eff.std(axis=0) / np.sqrt(eff.shape[0])
  ax_e.errorbar(range(1, 9), m, yerr=se, color='#0f3560', lw=2.0,
                marker='o', markersize=5, capsize=3)
  ax_e.set_xlabel('Member rank by mean sensitivity')
  ax_e.set_ylabel('Effective number\nof shocks')
  ax_e.set_ylim(0, 10)
  ax_e.set_xticks(range(1, 9))
  ax_e.spines['top'].set_visible(False)
  ax_e.spines['right'].set_visible(False)
  ax_e.text(-0.28, 1.10, 'b', transform=ax_e.transAxes,
            fontsize=17, fontweight='bold', color='#222222')

  # shared colorbar
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
  cax = fig.add_axes([0.925, 0.60, 0.013, 0.30])
  cbar = fig.colorbar(sm, cax=cax)
  cbar.set_label('Sensitivity to shock', fontsize=9.5)
  cbar.ax.tick_params(labelsize=8.5)

  name = f'fig-disks-eps{args.eps_label}'
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png (network {net}, {n_s}+{n_i})')


if __name__ == '__main__':
  main()
