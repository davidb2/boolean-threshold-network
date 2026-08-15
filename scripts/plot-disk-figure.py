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
  'font.size': 20,
  'mathtext.fontset': 'cm',
  'svg.fonttype': 'none',
})

SENS = '#ff7f0e'
INSENS = '#17becf'


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
    ax0.set_ylabel(label, fontsize=18, rotation=0, ha='right', va='center', labelpad=34)
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
  p.add_argument('--gallery', action='store_true',
                 help='all networks, one disk row each, sorted by composition')
  p.add_argument('--gallery-blocks', type=int, default=2)
  p.add_argument('--gallery-radial', action='store_true',
                 help='master circle: angle = network, radius = member rank')
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
  from matplotlib.colors import LinearSegmentedColormap
  cmap = LinearSegmentedColormap.from_list('whiteorange', ['#ffffff', '#ff7f0e', '#8c4a03'])
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.gallery_radial:
    rows = []
    for net, nodes in panels.items():
      if net not in snets:
        continue
      n_s = int((B[bnets.index(net), nodes] > cut).sum())
      rows.append((net, n_s, float(base.get(net, np.nan))))
    order = sorted(rows, key=lambda r: (-r[1], -(r[2] if r[2] == r[2] else 0)))
    n_nets = len(order)
    # sunflower geometry: disk diameter grows with radius so the innermost
    # ring sits close to the center with minimal empty hub
    c = 2 * np.sin(np.pi / n_nets)      # angular pitch as a fraction of radius
    grow = 1 + 0.97 * c                 # radial growth per rank
    r = np.array([grow ** k for k in range(8)])   # ring radii, r0 = 1
    d = 0.93 * c * r                    # disk diameters per ring
    E = r[-1] + d[-1]                   # half extent incl. rim label margin
    fig = plt.figure(figsize=(12.8, 13.4))
    cx, cy = 0.5, 0.53
    sx = 0.46 / E * (13.4 / 12.8)
    sy = 0.46 / E
    prev_ns = None
    for i, (net, n_s, acc) in enumerate(order):
      th = np.pi / 2 - 2 * np.pi * i / n_nets
      si, bi = snets.index(net), bnets.index(net)
      sens, insens = panel_groups(panels[net], B[bi], cut)
      for k, node in enumerate(sens + insens):
        x, y = r[k] * np.cos(th), r[k] * np.sin(th)
        ax = fig.add_axes([cx + x * sx - d[k] * sx / 2, cy + y * sy - d[k] * sy / 2,
                           d[k] * sx, d[k] * sy], projection='polar')
        draw_disk(ax, S[si][:, node], vmax, cmap, SENS if k < n_s else INSENS)
        ax.spines['polar'].set_linewidth(0.8 + 0.3 * k / 7)

    from matplotlib.patches import Wedge
    from matplotlib.colors import to_rgb
    ax_bg = fig.add_axes([cx - E * sx, cy - E * sy, 2 * E * sx, 2 * E * sy])
    ax_bg.set_xlim(-E, E); ax_bg.set_ylim(-E, E)
    ax_bg.set_aspect('equal'); ax_bg.axis('off')
    rgb_s, rgb_i = np.array(to_rgb(SENS)), np.array(to_rgb(INSENS))
    ring_out = 1 - 1.05 * d[0]
    ring_w = 0.15
    half = 180.0 / n_nets
    gap_deg = 2.2
    # one solid arc per composition group, with white gaps at each transition
    groups = []
    for i, (net, n_s, acc) in enumerate(order):
      if groups and groups[-1][0] == n_s:
        groups[-1][2] = i
      else:
        groups.append([n_s, i, i])
    for n_s, i0, i1 in groups:
      th_hi = 90 - 360.0 * i0 / n_nets + half - gap_deg / 2
      th_lo = 90 - 360.0 * i1 / n_nets - half + gap_deg / 2
      # intensity of red scales with the sensitive count, no blue mixing
      intensity = (n_s / 8.0) * 0.38
      color = intensity * rgb_s + (1 - intensity) * np.ones(3)
      ax_bg.add_patch(Wedge((0, 0), ring_out, th_lo, th_hi,
                            width=ring_w, facecolor=color, edgecolor='none'))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.28, 0.028, 0.44, 0.028])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Sensitivity to shock', fontsize=18)
    cbar.ax.tick_params(labelsize=17)
    fig.savefig(out_dir / 'fig-disks-radial.png', bbox_inches='tight', dpi=200)
    fig.savefig(out_dir / 'fig-disks-radial.svg', bbox_inches='tight')
    print(f'wrote {out_dir}/fig-disks-radial.png ({n_nets} networks)')
    return

  if args.gallery:
    rows = []
    for net, nodes in panels.items():
      if net not in snets:
        continue
      n_s = int((B[bnets.index(net), nodes] > cut).sum())
      rows.append((net, n_s, float(base.get(net, np.nan))))
    order = sorted(rows, key=lambda r: (-r[1], -(r[2] if r[2] == r[2] else 0)))
    nb = args.gallery_blocks
    per = int(np.ceil(len(order) / nb))
    blocks = [order[i * per:(i + 1) * per] for i in range(nb)]
    fig = plt.figure(figsize=(2.1 + 9.4 * nb * 0.62, 0.62 * per + 0.6))
    outer = fig.add_gridspec(1, nb, wspace=0.10)
    for b, blk in enumerate(blocks):
      grid = outer[b].subgridspec(per, 9, wspace=0.06, hspace=0.12)
      prev_ns = None
      for i, (net, n_s, acc) in enumerate(blk):
        si, bi = snets.index(net), bnets.index(net)
        sens, insens = panel_groups(panels[net], B[bi], cut)
        for k, node in enumerate(sens):
          ax = fig.add_subplot(grid[i, k], projection='polar')
          draw_disk(ax, S[si][:, node], vmax, cmap, SENS)
          ax.spines['polar'].set_linewidth(1.0)
        for k, node in enumerate(insens):
          ax = fig.add_subplot(grid[i, 9 - len(insens) + k], projection='polar')
          draw_disk(ax, S[si][:, node], vmax, cmap, INSENS)
          ax.spines['polar'].set_linewidth(1.0)
        if n_s != prev_ns:
          # the empty slot between the two groups carries the count
          ax_lab = fig.add_subplot(grid[i, n_s], frameon=False)
          ax_lab.set_xticks([]); ax_lab.set_yticks([])
          ax_lab.text(0.5, 0.5, str(n_s), transform=ax_lab.transAxes,
                      fontsize=24, color=SENS, ha='center', va='center',
                      fontweight='bold')
          prev_ns = n_s
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.92, 0.35, 0.008, 0.30])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Sensitivity to shock', fontsize=17)
    cbar.ax.tick_params(labelsize=15)
    fig.savefig(out_dir / 'fig-disks-gallery.png', bbox_inches='tight', dpi=200)
    fig.savefig(out_dir / 'fig-disks-gallery.svg', bbox_inches='tight')
    print(f'wrote {out_dir}/fig-disks-gallery.png ({len(order)} networks, {nb} blocks)')
    return

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

  fig = plt.figure(figsize=(12.8, 5.6))
  outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.55)
  gs_row = outer[0].subgridspec(1, 9, wspace=0.12)
  n_s, n_i = draw_panel_row(fig, gs_row, S[si], nodes, B[bi], cut, vmax, cmap)
  fig.text(0.06, 0.965, 'a', fontsize=31, fontweight='bold', color='#222222')
  # group labels centered under each group, computed from the actual axes
  fig.canvas.draw()
  pos = [ax.get_position() for ax in fig.axes if ax.name == 'polar']
  sens_axes, insens_axes = pos[:n_s], pos[n_s:n_s + n_i]
  if sens_axes:
    xc = 0.5 * (sens_axes[0].x0 + sens_axes[-1].x1)
    fig.text(xc, 0.53, 'sensitive members', fontsize=19, color=SENS, ha='center')
  if insens_axes:
    xc = 0.5 * (insens_axes[0].x0 + insens_axes[-1].x1)
    fig.text(xc, 0.53, 'insensitive members', fontsize=19, color=INSENS, ha='center')

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
  gs_b = outer[1].subgridspec(1, 10)
  ax_e = fig.add_subplot(gs_b[0, 3:8])
  m = eff.mean(axis=0)
  se = 1.96 * eff.std(axis=0) / np.sqrt(eff.shape[0])
  ax_e.errorbar(range(1, 9), m, yerr=se, color='#2ca02c', lw=2.0,
                marker='o', markersize=5, capsize=3)
  ax_e.set_xlabel('Member rank by mean sensitivity')
  ax_e.set_ylabel('Effective number\nof shocks', labelpad=8)
  ax_e.set_ylim(0, 10)
  ax_e.set_xticks(range(1, 9))
  ax_e.spines['top'].set_visible(False)
  ax_e.spines['right'].set_visible(False)
  ax_e.text(-0.42, 1.02, 'b', transform=ax_e.transAxes,
            fontsize=31, fontweight='bold', color='#222222')

  # shared colorbar
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
  cax = fig.add_axes([0.925, 0.60, 0.013, 0.30])
  cbar = fig.colorbar(sm, cax=cax)
  cbar.set_label('Sensitivity to shock', fontsize=18)
  cbar.ax.tick_params(labelsize=16)

  name = f'fig-disks-eps{args.eps_label}'
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png (network {net}, {n_s}+{n_i})')


if __name__ == '__main__':
  main()
