#!/usr/bin/env python3
'''Figure 1: methodology schematic (four panels for Illustrator assembly).

  a  directed Boolean threshold network + threshold update inset
  b  the same network under three shocks and a control
  c  state rasters from a real miniature simulation of the model
  d  the inference task: partial observation -> classifier -> shock label

Panels c and d use an actual simulation of the exact model (power-law
out-degree, U[-1,1] weights, threshold rule with ties keeping state,
shocks replacing outgoing weights of target nodes).

Usage:
  python scripts/plot-methods-figure.py --out-dir plots/fig1
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

BLUE_DARK = '#0f3560'
BLUE = '#1c5cab'
BLUE_MID = '#3987e5'
BLUE_SOFT = '#86b6ef'
EDGE_GRAY = '#8b93a1'
OFF = '#edf1f6'
SHOCK = '#d64541'
INK = '#333333'
MUTED = '#777777'

plt.rcParams.update({
  'font.size': 12,
  'svg.fonttype': 'none',
})


# ---------------------------------------------------------------------------
# miniature simulation of the exact model
# ---------------------------------------------------------------------------
def simulate(n=100, gamma=1.4, num_shocks=3, targets_per_shock=5, steps=40, rho=0.98, seed=3):
  rng = np.random.default_rng(seed)
  ks = np.arange(1, n + 1, dtype=np.float64)
  pk = ks ** (-gamma) / (ks ** (-gamma)).sum()
  W = np.zeros((n, n))
  for i in range(n):
    k = min(rng.choice(ks.astype(int), p=pk), n)
    tgts = rng.choice(n, size=k, replace=False)
    W[i, tgts] = rng.uniform(-1, 1, size=k)

  Ws = [W]
  shock_targets = []
  for _ in range(num_shocks):
    Wq = W.copy()
    tg = rng.choice(n, size=targets_per_shock, replace=False)
    for u in tg:
      mask = W[u] != 0
      Wq[u, mask] = rng.choice([-1.0, 1.0], size=mask.sum())
    Ws.append(Wq)
    shock_targets.append(tg)

  base = (rng.random(n) < 0.5).astype(np.int8)
  ic = np.where(rng.random(n) < rho, base, 1 - base).astype(np.int8)

  trajs = []
  for Wq in Ws:
    s = ic.copy()
    T = [s.copy()]
    for _ in range(steps):
      h = s @ Wq
      s = np.where(h > 0, 1, np.where(h < 0, 0, s)).astype(np.int8)
      T.append(s.copy())
    trajs.append(np.array(T).T)
  return trajs, shock_targets


# ---------------------------------------------------------------------------
# shared drawing helpers
# ---------------------------------------------------------------------------
NODE_POS = {
  0: (0.18, 0.78), 1: (0.46, 0.90), 2: (0.76, 0.82), 3: (0.92, 0.55),
  4: (0.72, 0.28), 5: (0.44, 0.12), 6: (0.16, 0.24), 7: (0.06, 0.52),
  8: (0.42, 0.52), 9: (0.66, 0.56),
}
NODE_STATE = {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
EDGES = [
  (0, 1, 0.8), (1, 2, -0.5), (2, 3, 0.9), (3, 9, -0.4), (9, 8, 0.7),
  (8, 0, -0.6), (7, 0, 0.5), (7, 8, 0.9), (6, 7, -0.7), (6, 5, 0.6),
  (5, 4, -0.8), (4, 3, 0.5), (4, 9, 0.6), (8, 5, -0.4), (1, 8, 0.55),
  (2, 9, -0.65),
]


def draw_network(ax, node_r=0.052, shock_target=None, dim=False):
  base_alpha = 0.35 if dim else 1.0
  for u, v, w in EDGES:
    x1, y1 = NODE_POS[u]
    x2, y2 = NODE_POS[v]
    shocked = shock_target is not None and u in shock_target
    color = SHOCK if shocked else EDGE_GRAY
    alpha = 1.0 if shocked else base_alpha
    lw = 0.8 + 2.2 * abs(w)
    style = '-|>' if w > 0 else '|-|,widthA=0,widthB=0.35'
    arrow = FancyArrowPatch(
      (x1, y1), (x2, y2),
      arrowstyle=style, mutation_scale=11,
      shrinkA=13, shrinkB=13,
      lw=lw, color=color, alpha=alpha,
      connectionstyle='arc3,rad=0.08', capstyle='round',
    )
    ax.add_patch(arrow)
  for j, (x, y) in NODE_POS.items():
    on = NODE_STATE[j] == 1
    is_target = shock_target is not None and j in shock_target
    face = BLUE if on else 'white'
    edge = SHOCK if is_target else (BLUE_DARK if on else EDGE_GRAY)
    ax.add_patch(Circle(
      (x, y), node_r, facecolor=face, edgecolor=edge,
      linewidth=2.2 if is_target else 1.4,
      alpha=1.0 if (is_target or not dim) else 0.75, zorder=5,
    ))


def bolt(ax, x, y, scale=1.0, color=SHOCK):
  pts = np.array([
    [0.35, 1.00], [0.75, 1.00], [0.50, 0.58], [0.80, 0.58],
    [0.18, 0.00], [0.38, 0.44], [0.10, 0.44],
  ])
  pts = (pts - [0.45, 0.5]) * 0.14 * scale + [x, y]
  ax.fill(pts[:, 0], pts[:, 1], color=color, zorder=8, lw=0)


def save(fig, out_dir, name):
  out_dir = pathlib.Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png')
  plt.close(fig)


# ---------------------------------------------------------------------------
# panel a (left): application vignettes that map onto the abstract model
# ---------------------------------------------------------------------------
def mini_net(ax, cx, cy, r=0.16, seed=2):
  rng = np.random.default_rng(seed)
  ang = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(0, 2 * np.pi)
  xs = cx + r * np.cos(ang)
  ys = cy + r * 0.8 * np.sin(ang)
  pairs = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
  for u, v in pairs:
    ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color=EDGE_GRAY, lw=1.0, alpha=0.8, zorder=3)
  for i, (x, y) in enumerate(zip(xs, ys)):
    on = i % 2 == 0
    ax.add_patch(Circle((x, y), 0.035, facecolor=BLUE if on else 'white',
                 edgecolor=BLUE_DARK if on else EDGE_GRAY, lw=1.0, zorder=4))


def tile(ax, label):
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal')
  ax.axis('off')
  ax.add_patch(FancyBboxPatch((0.03, 0.14), 0.94, 0.83,
               boxstyle='round,pad=0.01,rounding_size=0.05',
               facecolor='#f8fafc', edgecolor='#d5dbe3', lw=1.0, zorder=0))
  ax.text(0.5, 0.045, label, fontsize=11.5, ha='center', color=INK)


def vignette_bacterium(ax):
  tile(ax, 'gene regulation in bacteria')
  body = FancyBboxPatch((0.22, 0.40), 0.48, 0.30,
                        boxstyle='round,pad=0.02,rounding_size=0.14',
                        facecolor='#e7eef7', edgecolor=BLUE, lw=1.8, zorder=2)
  import matplotlib.transforms as mtransforms
  tr = mtransforms.Affine2D().rotate_deg_around(0.46, 0.55, -12) + ax.transData
  body.set_transform(tr)
  ax.add_patch(body)
  for y0 in [0.50, 0.58]:
    t = np.linspace(0, 1, 60)
    ax.plot(0.70 + 0.20 * t, y0 + 0.045 * np.sin(9 * t) - 0.12 * t * 0.3,
            color=BLUE, lw=1.3, zorder=1, transform=tr)
  mini_net(ax, 0.45, 0.555, r=0.135, seed=4)
  ax.add_patch(FancyBboxPatch((0.115, 0.76), 0.115, 0.055,
               boxstyle='round,pad=0.01,rounding_size=0.028',
               facecolor=SHOCK, edgecolor='none', zorder=4))
  ax.add_patch(FancyBboxPatch((0.175, 0.76), 0.056, 0.055,
               boxstyle='round,pad=0.01,rounding_size=0.028',
               facecolor='#f2b0ae', edgecolor='none', zorder=5))
  arr = FancyArrowPatch((0.235, 0.755), (0.335, 0.66), arrowstyle='-|>',
                        mutation_scale=9, lw=1.2, color=SHOCK, zorder=5)
  ax.add_patch(arr)
  ax.text(0.11, 0.885, 'drug', fontsize=10, color=SHOCK, ha='left')


def vignette_brain(ax):
  tile(ax, 'neural activity')
  pts = np.array([
    (0.335, 0.24), (0.315, 0.38), (0.295, 0.50), (0.30, 0.62), (0.335, 0.72),
    (0.40, 0.79), (0.48, 0.815), (0.56, 0.80), (0.615, 0.75), (0.645, 0.68),
    (0.655, 0.60), (0.645, 0.555), (0.685, 0.50), (0.695, 0.475), (0.655, 0.465),
    (0.665, 0.43), (0.645, 0.415), (0.655, 0.385), (0.625, 0.36), (0.60, 0.345),
    (0.565, 0.30), (0.545, 0.24),
  ])
  from scipy import interpolate
  tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0.0004, k=3)
  xs, ys = interpolate.splev(np.linspace(0, 1, 220), tck)
  ax.plot(xs, ys, color=BLUE, lw=1.8, zorder=2, solid_capstyle='round')
  mini_net(ax, 0.455, 0.60, r=0.115, seed=7)
  bolt(ax, 0.815, 0.755, scale=0.9)
  ax.text(0.815, 0.86, 'stimulus', fontsize=10, color=SHOCK, ha='center')
  arr = FancyArrowPatch((0.77, 0.71), (0.63, 0.645), arrowstyle='-|>',
                        mutation_scale=9, lw=1.2, color=SHOCK, zorder=5)
  ax.add_patch(arr)


def vignette_people(ax):
  tile(ax, 'human behaviour')
  pos = [(0.28, 0.62), (0.52, 0.72), (0.74, 0.58), (0.40, 0.38), (0.64, 0.34)]
  pairs = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
  for u, v in pairs:
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1] - 0.02, pos[v][1] - 0.02],
            color=EDGE_GRAY, lw=1.0, alpha=0.7, zorder=1)
  for i, (x, y) in enumerate(pos):
    on = i in (0, 2, 4)
    face = BLUE if on else 'white'
    edge = BLUE_DARK if on else EDGE_GRAY
    ax.add_patch(Circle((x, y + 0.045), 0.038, facecolor=face, edgecolor=edge, lw=1.2, zorder=3))
    body = FancyBboxPatch((x - 0.05, y - 0.075), 0.10, 0.085,
                          boxstyle='round,pad=0.01,rounding_size=0.045',
                          facecolor=face, edgecolor=edge, lw=1.2, zorder=3)
    ax.add_patch(body)
  bolt(ax, 0.14, 0.82, scale=0.9)
  ax.text(0.225, 0.875, 'disruption', fontsize=10, color=SHOCK, ha='left')


def panel_a_apps(out_dir):
  fig, axes = plt.subplots(3, 1, figsize=(3.1, 8.6))
  vignette_bacterium(axes[0])
  vignette_brain(axes[1])
  vignette_people(axes[2])
  save(fig, out_dir, 'fig1a-applications')


# ---------------------------------------------------------------------------
# panel a (right): network + threshold rule inset
# ---------------------------------------------------------------------------
def panel_a(out_dir):
  fig, (axn, axr) = plt.subplots(
    1, 2, figsize=(8.4, 3.7), gridspec_kw={'width_ratios': [1.05, 1.0]},
  )
  axn.set_xlim(-0.04, 1.02)
  axn.set_ylim(-0.03, 1.0)
  axn.set_aspect('equal')
  axn.axis('off')
  draw_network(axn)
  x8, y8 = NODE_POS[8]
  ring = Circle((x8, y8), 0.085, facecolor='none', edgecolor=INK, lw=1.0, ls=(0, (2, 2)), zorder=6)
  axn.add_patch(ring)
  leg_y = 0.0
  axn.add_patch(Circle((0.02, leg_y), 0.021, facecolor=BLUE, edgecolor=BLUE_DARK, lw=1.0))
  axn.text(0.06, leg_y, 'state 1', fontsize=10.5, color=INK, va='center')
  axn.add_patch(Circle((0.34, leg_y), 0.021, facecolor='white', edgecolor=EDGE_GRAY, lw=1.0))
  axn.text(0.38, leg_y, 'state 0', fontsize=10.5, color=INK, va='center')

  axr.set_xlim(0, 1)
  axr.set_ylim(0, 1)
  axr.axis('off')
  ins = [(0.80, 1, '$+0.9$'), (0.54, 0, '$+0.7$'), (0.28, 1, '$-0.4$')]
  for y, state, wlab in ins:
    axr.add_patch(Circle((0.08, y), 0.045,
                  facecolor=BLUE if state else 'white',
                  edgecolor=BLUE_DARK if state else EDGE_GRAY, lw=1.4))
    x_end, y_end = 0.315, 0.54 + (y - 0.54) * 0.22
    arrow = FancyArrowPatch(
      (0.13, y), (x_end, y_end),
      arrowstyle='-|>' if not wlab.startswith('$-') else '|-|,widthA=0,widthB=0.25',
      mutation_scale=10, lw=1.4, color=EDGE_GRAY, shrinkA=3, shrinkB=2,
    )
    axr.add_patch(arrow)
    axr.text((0.13 + x_end) / 2 - 0.01, (y + y_end) / 2 + 0.055, wlab,
             fontsize=10.5, color=MUTED, ha='center')
  axr.add_patch(FancyBboxPatch((0.33, 0.42), 0.35, 0.24,
                boxstyle='round,pad=0.02,rounding_size=0.03',
                facecolor='#f6f8fb', edgecolor=INK, lw=1.1))
  axr.text(0.505, 0.54, '$\\sum_i w_{ij}\\,\\sigma_i(t) > 0$ ?',
           fontsize=11, ha='center', va='center', color=INK)
  arrow = FancyArrowPatch((0.70, 0.54), (0.79, 0.54), arrowstyle='-|>', mutation_scale=11, lw=1.4, color=INK)
  axr.add_patch(arrow)
  axr.add_patch(Circle((0.86, 0.54), 0.05, facecolor=BLUE, edgecolor=BLUE_DARK, lw=1.4))
  axr.text(0.86, 0.40, '$\\sigma_j(t+1)$', fontsize=11, ha='center', color=INK)
  axr.text(0.03, 0.945, 'inputs at time $t$', fontsize=10.5, color=MUTED, ha='left')
  save(fig, out_dir, 'fig1a-network-and-rule')


# ---------------------------------------------------------------------------
# panel b: control + three shocks
# ---------------------------------------------------------------------------
def panel_b(out_dir):
  fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9))
  configs = [
    ('control', None),
    ('shock 1', {0}),
    ('shock 2', {4}),
    ('shock 3', {2, 6}),
  ]
  for ax, (label, tg) in zip(axes, configs):
    ax.set_xlim(-0.06, 1.04)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_network(ax, shock_target=tg, dim=tg is not None)
    if tg:
      for j in tg:
        x, y = NODE_POS[j]
        bolt(ax, x + 0.05, y + 0.09)
    color = INK if tg is None else SHOCK
    ax.set_title(label, fontsize=13, color=color, pad=4)
  save(fig, out_dir, 'fig1b-shocks')


# ---------------------------------------------------------------------------
# panel c: rasters from the real miniature simulation
# ---------------------------------------------------------------------------
def raster_rgb(traj, control, t0, order):
  on = np.array([15, 53, 96]) / 255
  off = np.array([237, 241, 246]) / 255
  red = np.array([214, 69, 65]) / 255
  sub = traj[order, t0:]
  ctl = control[order, t0:]
  img = np.where(sub[..., None] == 1, on, off)
  diff = sub != ctl
  img[diff] = 0.25 * img[diff] + 0.75 * red
  return img


def sensitivity_mini(trajs, t0=4):
  control = trajs[0]
  return np.mean(
    [np.mean(t[:, t0:] != control[:, t0:], axis=1) for t in trajs[1:]], axis=0,
  )


def pick_rows_and_reporters(trajs, n_rows=28, m=6, seed=5):
  rng = np.random.default_rng(seed)
  control = trajs[0]
  chosen = rng.choice(control.shape[0], size=n_rows, replace=False)
  order = chosen[np.argsort(-control[chosen].mean(axis=1), kind='stable')]
  reporters = list(rng.choice(order, size=m, replace=False))
  return order, reporters


def panel_c(out_dir, trajs, order, reporters, t0=4):
  control = trajs[0]
  fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
  labels = ['control', 'shock 1', 'shock 2', 'shock 3']
  for i, (ax, traj) in enumerate(zip(axes, trajs)):
    img = raster_rgb(traj, control, t0, order)
    ax.imshow(img, aspect='auto', interpolation='nearest')
    ax.set_title(labels[i], fontsize=13, color=INK if i == 0 else SHOCK, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
      s.set_color('#cccccc')
    if i == 0:
      ax.set_ylabel('nodes', fontsize=12, color=INK)
      rset = set(reporters)
      for row, node in enumerate(order):
        if node in rset:
          ax.plot(-1.6, row, marker='>', color=BLUE_MID, markersize=6, clip_on=False)
    ax.set_xlabel('time $\\rightarrow$', fontsize=12, color=INK)
  handles = [
    Rectangle((0, 0), 1, 1, facecolor=BLUE_DARK),
    Rectangle((0, 0), 1, 1, facecolor=OFF, edgecolor='#cccccc'),
    Rectangle((0, 0), 1, 1, facecolor='#c96f6c'),
  ]
  fig.legend(
    handles, ['state 1', 'state 0', 'differs from control'],
    loc='lower center', ncol=3, frameon=False, fontsize=10.5,
    bbox_to_anchor=(0.5, -0.06),
  )
  save(fig, out_dir, 'fig1c-trajectories')


# ---------------------------------------------------------------------------
# panel d: partial observation -> classifier -> label
# ---------------------------------------------------------------------------
def panel_d(out_dir, trajs, reporters, t0=25, true_shock=2):
  control = trajs[0]
  m = len(reporters)
  reporters = np.array(reporters)

  fig, ax = plt.subplots(figsize=(9.6, 3.2))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis('off')

  img = raster_rgb(trajs[true_shock], control, t0, reporters)
  ax.imshow(
    img, aspect='auto', interpolation='nearest',
    extent=(0.04, 0.40, 0.30, 0.78), zorder=3,
  )
  ax.add_patch(Rectangle((0.04, 0.30), 0.36, 0.48, fill=False, edgecolor='#cccccc', lw=1.0, zorder=4))
  ax.text(0.22, 0.86, f'$m={m}$ reporters observed', fontsize=12, ha='center', color=INK)
  ax.text(0.22, 0.20, 'one noisy trial, late times', fontsize=11, ha='center', color=MUTED)

  arrow = FancyArrowPatch((0.43, 0.54), (0.52, 0.54), arrowstyle='-|>', mutation_scale=13, lw=1.6, color=INK)
  ax.add_patch(arrow)

  ax.add_patch(FancyBboxPatch((0.53, 0.42), 0.17, 0.24,
               boxstyle='round,pad=0.02,rounding_size=0.03',
               facecolor='#f6f8fb', edgecolor=INK, lw=1.2))
  ax.text(0.615, 0.575, 'classifier', fontsize=12.5, ha='center', color=INK)
  ax.text(0.615, 0.47, 'which shock?', fontsize=10.5, ha='center', color=MUTED)

  arrow = FancyArrowPatch((0.72, 0.54), (0.80, 0.54), arrowstyle='-|>', mutation_scale=13, lw=1.6, color=INK)
  ax.add_patch(arrow)

  options = ['control', 'shock 1', 'shock 2', 'shock 3']
  ys = [0.78, 0.62, 0.46, 0.30]
  for lab, y in zip(options, ys):
    chosen = lab == f'shock {true_shock}'
    ax.text(
      0.84, y, lab, fontsize=12,
      color='white' if chosen else MUTED,
      fontweight='bold' if chosen else 'normal',
      bbox=dict(
        boxstyle='round,pad=0.32',
        facecolor=SHOCK if chosen else '#f0f2f5',
        edgecolor=SHOCK if chosen else '#d5d9df',
        lw=1.0,
      ),
      ha='left', va='center',
    )
  save(fig, out_dir, 'fig1d-inference')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()
  trajs, shock_targets = simulate()
  print('shock targets:', shock_targets)
  panel_a_apps(args.out_dir)
  panel_a(args.out_dir)
  panel_b(args.out_dir)
  order, reporters = pick_rows_and_reporters(trajs)
  panel_c(args.out_dir, trajs, order, reporters)
  panel_d(args.out_dir, trajs, reporters)


if __name__ == '__main__':
  main()
