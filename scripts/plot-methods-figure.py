#!/usr/bin/env python3
'''Figure 1: methodology schematic, one file per panel.

  fig1b-rule           the threshold update rule
  fig1c-shock-closeup  one shock: the outgoing weights of a target node
  fig1d-dynamics       a control trajectory and 2 shocked copies of it
  fig1e-noise          what noise does to the initial condition
  fig1f-inference      the reporter panel and the 3 alternatives

Panel a is a BioRender export and is not drawn here. Node states are
black when active and white when inactive, weights are shaded from white
at magnitude 0 to black at magnitude 1, amber marks shock 1 and pink
shock 2. Panels d to f run the exact update rule on a real network of 10
nodes, picked by a seed search for one readable example.

Usage:
  python scripts/plot-methods-figure.py --out-dir plots/fig1
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, Wedge)

BLUE = '#1c5cab'        # state 1 fill
BLUE_DARK = '#0f3560'
POS = '#3987e5'         # positive weight
NEG = '#e34948'         # negative weight
AMBER = '#e8a000'       # shock accent, shock 1
PINK = '#e0559b'        # shock 2
CREAM = '#f6efdb'       # control slice of the wheel
OFF = '#eef1f6'         # state 0 raster cell
INK = '#333333'
MUTED = '#777777'
TILE_BG = '#f8fafc'
TILE_EDGE = '#d5dbe3'

plt.rcParams.update({
  'font.size': 12,
  'font.family': 'serif',
  'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
  'mathtext.fontset': 'stix',
  'svg.fonttype': 'none',
})


# ---------------------------------------------------------------------------
# miniature simulation of the exact model
# ---------------------------------------------------------------------------
def run(W, ic, steps):
  """The exact update rule: a node turns on when its weighted input is
  positive, off when it is negative, and holds when the sum is zero."""
  s = np.asarray(ic).copy()
  T = [s.copy()]
  for _ in range(steps):
    h = s @ W
    s = np.where(h > 0, 1, np.where(h < 0, 0, s)).astype(np.int8)
    T.append(s.copy())
  return np.array(T).T


def small_network(n=10, num_shocks=2, targets_per_shock=2, steps=24, seed=0):
  """A real network of n nodes with its shocked copies. Each node sends 2
  to 4 edges with weights drawn uniformly on minus one to one, and a shock
  redraws every outgoing weight of its targets to plus or minus one."""
  rng = np.random.default_rng(seed)
  W = np.zeros((n, n))
  for i in range(n):
    k = int(rng.integers(2, 5))
    tgts = rng.choice([j for j in range(n) if j != i], size=k, replace=False)
    W[i, tgts] = rng.uniform(-1, 1, size=k)
  Ws, targets = [W], []
  for _ in range(num_shocks):
    Wq = W.copy()
    tg = rng.choice(n, size=targets_per_shock, replace=False)
    for u in tg:
      mask = W[u] != 0
      Wq[u, mask] = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    Ws.append(Wq)
    targets.append(tg)
  ic = (rng.random(n) < 0.5).astype(np.int8)
  return Ws, ic, targets, [run(Wq, ic, steps) for Wq in Ws]


def small_sim(t_max=14, n_copies=7, eps=0.30, m=4, n_snap=7, true_shock=2,
              steps=24, seed_max=6000):
  """Search seeds for one clear example: both shocked copies drift away
  from the control at a readable rate, they differ from each other, and
  the control itself keeps moving over the window that is drawn."""
  for seed in range(seed_max):
    Ws, ic, targets, trajs = small_network(steps=steps, seed=seed)
    c, s1, s2 = [t[:, :t_max] for t in trajs]
    d1, d2 = (s1 != c).mean(), (s2 != c).mean()
    if not (0.13 < d1 < 0.42 and 0.13 < d2 < 0.42):
      continue
    if (s1 != s2).mean() < 0.12 or c.std(axis=1).mean() < 0.16:
      continue
    # the control must still be moving late in the window, so the picture
    # is not a frozen state repeated across the strip
    late = c[:, t_max // 2:]
    if sum((late[:, t] != late[:, t + 1]).any()
           for t in range(late.shape[1] - 1)) < 2:
      continue
    if (s1[:, -1] != c[:, -1]).sum() < 2 or (s2[:, -1] != c[:, -1]).sum() < 2:
      continue
    rng = np.random.default_rng(seed + 991)
    noisy = np.stack([np.where(rng.random(len(ic)) < eps / 2, 1 - ic, ic)
                      for _ in range(n_copies)], axis=1).astype(np.int8)
    flips = (noisy != ic[:, None]).sum(axis=0)
    if flips.min() < 1 or flips.max() > 3:
      continue
    # one noisy trial of the true shock, read at the most informative window
    trial = run(Ws[true_shock], noisy[:, 0], steps)
    best = max(range(steps - n_snap), key=lambda t: np.sort(
        trial[:, t:t + n_snap].std(axis=1))[-m:].sum())
    win = trial[:, best:best + n_snap]
    rep = np.sort(np.argsort(-win.std(axis=1))[:m])
    if win[rep].std(axis=1).min() == 0:
      continue
    print(f'panels d to f: seed {seed}, shock targets {targets}, '
          f'drift {d1:.2f} and {d2:.2f}, noise flips {list(flips)}, '
          f'reporters {list(rep)} from step {best}')
    return dict(trajs=trajs, ic=ic, noisy=noisy, observed=win[rep],
                targets=targets, seed=seed)
  raise SystemExit('no seed met the criteria')


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


def edge_color(w):
  return POS if w > 0 else NEG


def draw_network(ax, node_r=0.052, target=None, dim=False, focal_ring=None):
  base_alpha = 0.30 if dim else 0.95
  for u, v, w in EDGES:
    x1, y1 = NODE_POS[u]
    x2, y2 = NODE_POS[v]
    from_target = target is not None and u == target
    alpha = 1.0 if from_target else base_alpha
    arrow = FancyArrowPatch(
      (x1, y1), (x2, y2),
      arrowstyle='-|>', mutation_scale=10,
      shrinkA=13, shrinkB=13,
      lw=0.7 + 2.4 * abs(w), color=edge_color(w), alpha=alpha,
      connectionstyle='arc3,rad=0.08', capstyle='round', zorder=2,
    )
    ax.add_patch(arrow)
  for j, (x, y) in NODE_POS.items():
    on = NODE_STATE[j] == 1
    is_target = target is not None and j == target
    face = BLUE if on else 'white'
    edge = AMBER if is_target else (BLUE_DARK if on else '#9aa5b1')
    ax.add_patch(Circle(
      (x, y), node_r, facecolor=face, edgecolor=edge,
      linewidth=2.4 if is_target else 1.4,
      alpha=1.0 if (is_target or not dim) else 0.8, zorder=5,
    ))
  if focal_ring is not None:
    x, y = NODE_POS[focal_ring]
    ax.add_patch(Circle((x, y), 0.085, facecolor='none', edgecolor=INK,
                 lw=1.0, ls=(0, (2, 2)), zorder=6))


def squiggle(ax, x0, x1, y, amp=0.85, cycles=1.75, lw=2.4, color=INK):
  """A wavy arrow, drawn as a windowed sine that lands flat, with a short
  straight head so the point stays sharp."""
  t = np.linspace(0, 1, 240)
  x = x0 + (x1 - 0.85 - x0) * t
  yy = y + amp * np.sin(np.pi * t) * np.sin(2 * np.pi * cycles * t)
  ax.plot(x, yy, color=color, lw=lw, solid_capstyle='round', zorder=3)
  ax.add_patch(FancyArrowPatch((x1 - 0.88, y), (x1, y), arrowstyle='-|>',
               mutation_scale=16, lw=lw, color=color, shrinkA=0, shrinkB=0,
               joinstyle='miter', capstyle='butt', zorder=3))


def bolt(ax, x, y, scale=1.0, color=AMBER):
  pts = np.array([
    [0.35, 1.00], [0.75, 1.00], [0.50, 0.58], [0.80, 0.58],
    [0.18, 0.00], [0.38, 0.44], [0.10, 0.44],
  ])
  pts = (pts - [0.45, 0.5]) * 0.14 * scale + [x, y]
  ax.fill(pts[:, 0], pts[:, 1], color=color, zorder=8, lw=0, clip_on=False)


def save(fig, out_dir, name):
  out_dir = pathlib.Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight')
  fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out_dir}/{name}.svg + .png')
  plt.close(fig)


# ---------------------------------------------------------------------------
# panel a: application vignettes (horizontal row)
# ---------------------------------------------------------------------------
def mini_net(ax, cx, cy, r=0.16, seed=2):
  rng = np.random.default_rng(seed)
  ang = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(0, 2 * np.pi)
  xs = cx + r * np.cos(ang)
  ys = cy + r * 0.8 * np.sin(ang)
  pairs = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
  for u, v in pairs:
    ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color='#9aa5b1', lw=1.0, alpha=0.8, zorder=3)
  for i, (x, y) in enumerate(zip(xs, ys)):
    on = i % 2 == 0
    ax.add_patch(Circle((x, y), 0.035, facecolor=BLUE if on else 'white',
                 edgecolor=BLUE_DARK if on else '#9aa5b1', lw=1.0, zorder=4))


def tile(ax, label):
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal')
  ax.axis('off')
  ax.add_patch(FancyBboxPatch((0.03, 0.14), 0.94, 0.83,
               boxstyle='round,pad=0.01,rounding_size=0.05',
               facecolor=TILE_BG, edgecolor=TILE_EDGE, lw=1.0, zorder=0))
  ax.text(0.5, 0.045, label, fontsize=11.5, ha='center', color=INK)


def vignette_bacterium(ax):
  tile(ax, 'gene regulation in bacteria')
  body = FancyBboxPatch((0.22, 0.40), 0.48, 0.30,
                        boxstyle='round,pad=0.02,rounding_size=0.14',
                        facecolor='#e7eef7', edgecolor=BLUE, lw=1.8, zorder=2)
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
               facecolor=AMBER, edgecolor='none', zorder=4))
  ax.add_patch(FancyBboxPatch((0.175, 0.76), 0.056, 0.055,
               boxstyle='round,pad=0.01,rounding_size=0.028',
               facecolor='#f5d68d', edgecolor='none', zorder=5))
  arr = FancyArrowPatch((0.235, 0.755), (0.335, 0.66), arrowstyle='-|>',
                        mutation_scale=9, lw=1.2, color=AMBER, zorder=5)
  ax.add_patch(arr)
  ax.text(0.11, 0.885, 'drug', fontsize=10, color=AMBER, ha='left')


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
  ax.text(0.815, 0.86, 'stimulus', fontsize=10, color=AMBER, ha='center')
  arr = FancyArrowPatch((0.77, 0.71), (0.63, 0.645), arrowstyle='-|>',
                        mutation_scale=9, lw=1.2, color=AMBER, zorder=5)
  ax.add_patch(arr)


def vignette_people(ax):
  tile(ax, 'human behaviour')
  pos = [(0.28, 0.62), (0.52, 0.72), (0.74, 0.58), (0.40, 0.38), (0.64, 0.34)]
  pairs = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
  for u, v in pairs:
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1] - 0.02, pos[v][1] - 0.02],
            color='#9aa5b1', lw=1.0, alpha=0.7, zorder=1)
  for i, (x, y) in enumerate(pos):
    on = i in (0, 2, 4)
    face = BLUE if on else 'white'
    edge = BLUE_DARK if on else '#9aa5b1'
    ax.add_patch(Circle((x, y + 0.045), 0.038, facecolor=face, edgecolor=edge, lw=1.2, zorder=3))
    body = FancyBboxPatch((x - 0.05, y - 0.075), 0.10, 0.085,
                          boxstyle='round,pad=0.01,rounding_size=0.045',
                          facecolor=face, edgecolor=edge, lw=1.2, zorder=3)
    ax.add_patch(body)
  bolt(ax, 0.14, 0.82, scale=0.9)
  ax.text(0.225, 0.875, 'disruption', fontsize=10, color=AMBER, ha='left')


def panel_a_apps(out_dir):
  fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
  vignette_bacterium(axes[0])
  vignette_brain(axes[1])
  vignette_people(axes[2])
  save(fig, out_dir, 'fig1a-applications')


# ---------------------------------------------------------------------------
# panel b: the threshold update rule
# ---------------------------------------------------------------------------
def signed_arc(ax, p0, p1, mag, sign, rad=0.0, lw=3.2, bar=0.030, head=15):
  """A gently arched edge in one flat color for stem, head, and border.
  The head is a sharp triangle for positive weights and a perpendicular
  bar for negative ones; every edge has the same stem width and head
  size, and the gray level carries the weight magnitude."""
  p0 = np.asarray(p0, float)
  p1 = np.asarray(p1, float)
  fc = (1 - mag, 1 - mag, 1 - mag)
  if sign >= 0:
    arrow = FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                            arrowstyle='-|>', mutation_scale=head, lw=lw,
                            facecolor=fc, edgecolor=fc, shrinkA=0, shrinkB=0,
                            capstyle='butt', joinstyle='miter', zorder=3)
    ax.add_patch(arrow)
  else:
    arrow = FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                            arrowstyle='-', lw=lw, facecolor=fc, edgecolor=fc,
                            shrinkA=0, shrinkB=0, capstyle='butt', zorder=3)
    ax.add_patch(arrow)
    d = p1 - p0
    L = np.hypot(*d)
    n = np.array([-d[1], d[0]]) / L
    ctrl = (p0 + p1) / 2 + rad * L * n
    t = p1 - ctrl
    t = t / np.hypot(*t)
    nt = np.array([-t[1], t[0]])
    b = bar
    ax.plot([p1[0] - b * nt[0], p1[0] + b * nt[0]],
            [p1[1] - b * nt[1], p1[1] + b * nt[1]],
            color=fc, lw=lw + 2.2, solid_capstyle='butt', zorder=3)


def panel_b_rule(out_dir):
  """The threshold mechanism alone. Input states are black (active) or
  white (inactive); edges are identical in size, shaded by weight
  magnitude, with a bar head for repression. Arrows hover on an invisible
  buffer around the box and around each node, and the gauge needle sits
  just past the dashed threshold, so the target node activates."""
  fig, axr = plt.subplots(figsize=(6.4, 4.6))
  axr.set_xlim(0, 1.30)
  axr.set_ylim(0.02, 0.99)
  axr.set_aspect('equal')
  axr.axis('off')

  # active weights sum to +0.15, so the needle sits just past threshold
  ins = [(0.865, 1, 0.85), (0.678, 0, -0.45), (0.490, 1, -0.50),
         (0.303, 0, 0.30), (0.115, 1, -0.20)]
  r_node, gap_node = 0.042, 0.020
  box_c = np.array([0.665, 0.48])
  r_field = 0.27
  # tips sit at evenly spaced angles on the field circle, so the arrows
  # rest on the buffer without touching one another
  for i, (y, state, w) in enumerate(ins):
    c = np.array([0.075, y])
    axr.add_patch(Circle(c, r_node,
                  facecolor='#1a1a1a' if state else 'white',
                  edgecolor='#000000', lw=1.4))
    theta = np.deg2rad(146 + 17 * i)
    p1 = box_c + r_field * np.array([np.cos(theta), np.sin(theta)])
    u = p1 - c
    u = u / np.hypot(*u)
    p0 = c + u * (r_node + gap_node)
    rad = -0.20 * (y - 0.50)
    signed_arc(axr, p0, p1, abs(w), np.sign(w), rad=rad)

  axr.add_patch(FancyBboxPatch((0.49, 0.36), 0.35, 0.24,
                boxstyle='round,pad=0.02,rounding_size=0.03',
                facecolor=TILE_BG, edgecolor=INK, lw=1.1))
  cx, cy, R = 0.665, 0.40, 0.16
  from matplotlib.patches import Wedge
  axr.add_patch(Wedge((cx, cy), R, 0, 180, facecolor='white',
                      edgecolor=INK, lw=1.3, zorder=4))
  axr.plot([cx, cx], [cy, cy + R * 0.97], color='#8c8c8c', lw=1.3,
           linestyle=(0, (3.2, 2.6)), zorder=5)
  ang = np.deg2rad(66)
  axr.plot([cx, cx + 0.90 * R * np.cos(ang)], [cy, cy + 0.90 * R * np.sin(ang)],
           color='#c1272d', lw=2.4, zorder=6, solid_capstyle='butt')
  axr.add_patch(Circle((cx, cy), 0.015, facecolor='#c1272d',
                       edgecolor='none', zorder=7))

  out_arrow = FancyArrowPatch((0.885, 0.48), (1.045, 0.48), arrowstyle='-|>',
                              mutation_scale=16, lw=3.2, facecolor='#1a1a1a',
                              edgecolor='#1a1a1a', shrinkA=0, shrinkB=0,
                              capstyle='butt', joinstyle='miter', zorder=3)
  axr.add_patch(out_arrow)
  axr.add_patch(Circle((1.11, 0.48), 0.05, facecolor='#1a1a1a',
                       edgecolor='#000000', lw=1.4))
  # state key, tucked into the empty corner under the output node
  for y, state, label in [(0.205, 1, 'active'), (0.095, 0, 'inactive')]:
    axr.add_patch(Circle((0.90, y), r_node,
                  facecolor='#1a1a1a' if state else 'white',
                  edgecolor='#000000', lw=1.4))
    axr.text(0.955, y, label, fontsize=11, color=INK, va='center', ha='left')
  save(fig, out_dir, 'fig1b-rule')


# ---------------------------------------------------------------------------
# panels d to f: real dynamics, noise, and the inference task
# ---------------------------------------------------------------------------
def circle_cells(ax, states, x0, y0, cell=1.0, ref=None, edge='#000000',
                 r=0.36, lw=0.7):
  """A block of node states as circles, filled for active and open for
  inactive. Where a state differs from the reference block, a thin
  diagonal runs through the circle in the opposite color."""
  n_rows, n_cols = states.shape
  for i in range(n_rows):
    for j in range(n_cols):
      cx = x0 + (j + 0.5) * cell
      cy = y0 + (n_rows - 1 - i + 0.5) * cell
      on = states[i, j] == 1
      ax.add_patch(Circle((cx, cy), r * cell,
                   facecolor='#1a1a1a' if on else 'white',
                   edgecolor=edge, lw=lw, zorder=2))
      if ref is not None and states[i, j] != ref[i, j]:
        d = 0.66 * r * cell
        ax.plot([cx - d, cx + d], [cy - d, cy + d],
                color='white' if on else '#1a1a1a', lw=lw + 0.2,
                solid_capstyle='round', zorder=3)


def time_arrow(ax, x0, x1, y, label='time', fontsize=11, lw=1.3):
  ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
               mutation_scale=12, lw=lw, color=INK, shrinkA=0, shrinkB=0,
               joinstyle='miter', capstyle='butt'))
  ax.text(x1 + 0.4, y, label, fontsize=fontsize, color=INK, va='center')


def panel_d(out_dir, sim, t_max=14):
  """One control trajectory and 2 shocked copies of it, all from the same
  initial condition, for a real network of 10 nodes."""
  ctl, s1, s2 = [t[:, :t_max] for t in sim['trajs']]
  n_rows = ctl.shape[0]
  fig, ax = plt.subplots(figsize=(3.3, 8.0))
  gap = 3.0
  rows = [('control', INK, '#000000', ctl, None),
          ('shock 1', AMBER, AMBER, s1, ctl),
          ('shock 2', PINK, PINK, s2, ctl)]
  for k, (label, tc, ec, states, ref) in enumerate(rows):
    y0 = (len(rows) - 1 - k) * (n_rows + gap)
    circle_cells(ax, states, 0.0, y0, ref=ref, edge=ec)
    ax.text(0.0, y0 + n_rows + 0.55, label, fontsize=13, color=tc,
            ha='left', va='bottom')
  time_arrow(ax, 0.0, t_max * 0.42, -2.1)
  ax.text(-1.5, (len(rows) * (n_rows + gap) - gap) / 2, 'nodes', fontsize=11,
          color=INK, rotation=90, ha='center', va='center')
  ax.set_xlim(-2.2, t_max + 0.4)
  ax.set_ylim(-3.1, len(rows) * (n_rows + gap) - gap + 1.9)
  ax.set_aspect('equal')
  ax.axis('off')
  save(fig, out_dir, 'fig1d-dynamics')


def panel_e(out_dir, sim, n_copies=7, spacing=2.85):
  """What noise does to the initial condition: the same starting state,
  then noisy copies of it, with a diagonal on every node that flipped."""
  ic = sim['ic'].reshape(-1, 1)
  copies = sim['noisy'][:, :n_copies]
  n_rows = ic.shape[0]
  fig, ax = plt.subplots(figsize=(8.4, 4.0))
  circle_cells(ax, ic, 0.0, 0.0, edge='#000000', lw=1.0)
  x_first = 5.6
  squiggle(ax, 1.5, x_first - 0.55, n_rows / 2)
  for k in range(n_copies):
    circle_cells(ax, copies[:, k:k + 1], x_first + k * spacing, 0.0,
                 ref=ic, edge='#000000', lw=1.0)
  ax.set_xlim(-0.4, x_first + (n_copies - 1) * spacing + 1.4)
  ax.set_ylim(-0.5, n_rows + 0.5)
  ax.set_aspect('equal')
  ax.axis('off')
  save(fig, out_dir, 'fig1e-noise')


def panel_f(out_dir, sim, n_snap=7):
  """The inference task: a small reporter panel watched for a few steps in
  one noisy trial, and 3 alternatives to decide between."""
  obs = sim['observed'][:, :n_snap]
  m = obs.shape[0]
  fig, ax = plt.subplots(figsize=(8.6, 3.2))
  circle_cells(ax, obs, 0.0, 0.0, edge='#000000', lw=1.0)
  ax.text(0.0, m + 0.7, f'Reporter panel consisting of {m} members',
          fontsize=13, color=INK, ha='left', va='bottom')
  time_arrow(ax, 0.0, n_snap * 0.5, -1.5)

  x_ar = n_snap + 1.1
  ax.add_patch(FancyArrowPatch((x_ar, m / 2), (x_ar + 3.0, m / 2),
               arrowstyle='-|>', mutation_scale=26, lw=3.0, color=INK,
               shrinkA=0, shrinkB=0, joinstyle='miter', capstyle='butt'))

  cx, cy, R = x_ar + 6.6, m / 2, 3.1
  for k, color in enumerate([CREAM, AMBER, PINK]):
    ax.add_patch(Wedge((cx, cy), R, 90 + 120 * k, 210 + 120 * k,
                 facecolor=color, edgecolor='#111111', lw=1.0, zorder=2))
  ax.add_patch(Circle((cx, cy), 0.42 * R, facecolor='white',
                      edgecolor='#111111', lw=1.0, zorder=3))
  ax.text(cx, cy, '?', fontsize=34, color=INK, ha='center', va='center',
          zorder=4)
  ax.set_xlim(-0.4, cx + R + 0.5)
  ax.set_ylim(min(-2.1, cy - R - 0.4), max(m + 1.7, cy + R + 0.4))
  ax.set_aspect('equal')
  ax.axis('off')
  save(fig, out_dir, 'fig1f-inference')


# ---------------------------------------------------------------------------
# panel c: one shock, the outgoing weights of a target node
# ---------------------------------------------------------------------------
def out_star(ax, c, weights, r_field=0.44, r_node=0.072, gap=0.045,
             phase=30.0, lw=3.0, head=17, bar=0.038, ring=INK):
  """One node with its outgoing edges, drawn in the same language as the
  update rule: identical edge length and stem width, gray level carrying
  the magnitude, pointed head for activation and a bar for repression.
  The edges leave in every direction, evenly spaced around the node, and
  each one stands off the perimeter on the same gap."""
  c = np.asarray(c, float)
  n = len(weights)
  for i, w in enumerate(weights):
    th = np.deg2rad(phase + 360.0 * i / n)
    u = np.array([np.cos(th), np.sin(th)])
    signed_arc(ax, c + (r_node + gap) * u, c + r_field * u, abs(w),
               np.sign(w), rad=0.12, lw=lw, bar=bar, head=head)
  ax.add_patch(Circle(c, r_node, facecolor='#1a1a1a', edgecolor=ring,
                      lw=0.65 * lw, zorder=5))


def panel_c(out_dir):
  """The shock, close up. The same active node twice, with the same six
  outgoing edges: mixed signs and magnitudes before, all of magnitude one
  after, some of them with their sign switched."""
  fig, ax = plt.subplots(figsize=(9.6, 3.4))
  ax.set_xlim(0.04, 2.33)
  ax.set_ylim(0.0, 1.00)
  ax.set_aspect('equal')
  ax.axis('off')

  before = [0.85, -0.30, 0.45, -0.65, 0.15, -0.55]
  after = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]

  # the composed figure shows this panel at about half the size it is
  # drawn at, so everything measured in points is drawn twice as large
  s = 2.0
  ca, cb = np.array([0.50, 0.50]), np.array([1.87, 0.50])
  out_star(ax, ca, before, lw=2.5 * s, head=10 * s, bar=0.048,
           ring='#000000')
  out_star(ax, cb, after, lw=2.5 * s, head=10 * s, bar=0.048, ring='#000000')

  # the shock sits centered in the gap between the two stars
  mid = 0.5 * (ca[0] + cb[0])
  half = 0.24
  bolt(ax, mid, 0.855, scale=1.6)
  arrow = FancyArrowPatch((mid - half, 0.46), (mid + half, 0.46),
                          arrowstyle='-|>', mutation_scale=26 * s, lw=4.5 * s,
                          color=AMBER, shrinkA=0, shrinkB=0,
                          joinstyle='miter', capstyle='butt')
  ax.add_patch(arrow)
  ax.text(mid, 0.585, 'shock', fontsize=13 * s, color=AMBER, ha='center')
  save(fig, out_dir, 'fig1c-shock-closeup')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()
  panel_b_rule(args.out_dir)
  panel_c(args.out_dir)
  sim = small_sim()
  panel_d(args.out_dir, sim)
  panel_e(args.out_dir, sim)
  panel_f(args.out_dir, sim)


if __name__ == '__main__':
  main()
