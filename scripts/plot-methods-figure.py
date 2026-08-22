#!/usr/bin/env python3
'''Figure 1: methodology schematic (horizontal layout, five panel files).

  fig1a-applications      three application vignettes (row)
  fig1a-network-and-rule  the general model: network + threshold update
  fig1b-dynamics          Boolean dynamics over time, control vs shocked
  fig1c-shock-closeup     one concrete shock: outgoing weights before/after
  fig1d-inference         the identification problem

Color code: blue edges = positive weights, red edges = negative weights,
thickness = |w|. Amber = the shock/perturbation. Node fill = state.
Panels b and d use a real miniature simulation of the exact model.

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

BLUE = '#1c5cab'        # state 1 fill
BLUE_DARK = '#0f3560'
POS = '#3987e5'         # positive weight
NEG = '#e34948'         # negative weight
AMBER = '#e8a000'       # shock accent
OFF = '#eef1f6'         # state 0 raster cell
INK = '#333333'
MUTED = '#777777'
TILE_BG = '#f8fafc'
TILE_EDGE = '#d5dbe3'

plt.rcParams.update({
  'font.size': 12,
  'mathtext.fontset': 'cm',
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
# chunky state strips (nodes x time) with visible grid
# ---------------------------------------------------------------------------
def draw_strip(ax, states, diff=None, cell=1.0):
  n_rows, n_cols = states.shape
  for r in range(n_rows):
    for c in range(n_cols):
      on = states[r, c] == 1
      face = BLUE if on else OFF
      ec = 'white'
      ax.add_patch(Rectangle((c * cell, (n_rows - 1 - r) * cell), cell, cell,
                   facecolor=face, edgecolor=ec, linewidth=1.1, zorder=2))
      if diff is not None and diff[r, c]:
        ax.add_patch(Rectangle((c * cell + 0.14, (n_rows - 1 - r) * cell + 0.14),
                     cell - 0.28, cell - 0.28,
                     facecolor='none', edgecolor=AMBER, linewidth=1.7, zorder=3))
  ax.set_xlim(-0.3, n_cols * cell + 0.3)
  ax.set_ylim(-0.3, n_rows * cell + 0.3)
  ax.set_aspect('equal')
  ax.axis('off')


def pick_display_nodes(trajs, shock_idx=1, n_show=12, t_max=16, seed=5):
  rng = np.random.default_rng(seed)
  ctl = trajs[0][:, :t_max]
  shk = trajs[shock_idx][:, :t_max]
  diverge = (ctl != shk).mean(axis=1)
  osc = ctl.std(axis=1)
  strong = np.argsort(-diverge)[:20]
  lively = np.argsort(-osc)[:30]
  static = np.where(osc == 0)[0]
  chosen = []
  chosen += list(rng.choice(strong, 5, replace=False))
  chosen += [n for n in rng.choice(lively, 10, replace=False) if n not in chosen][:4]
  chosen += [n for n in rng.choice(static, 5, replace=False) if n not in chosen][:3]
  chosen = np.array(chosen[:n_show])
  first_div = np.argmax(np.concatenate([ctl[chosen] != shk[chosen],
                        np.ones((len(chosen), 1), bool)], axis=1), axis=1)
  return chosen[np.argsort(first_div)]


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
# panel a (right): network + threshold rule inset
# ---------------------------------------------------------------------------
def signed_arrow(ax, p0, p1, mag, sign, stem=0.011):
  """A flat-color arrow with a thin black border. Magnitude sets the fill
  on a white to black ramp, the head is a sharp triangle for positive
  weights and a perpendicular bar for negative ones, and every arrow has
  the same stem width and head size."""
  p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
  d = p1 - p0
  u = d / np.hypot(*d)
  n = np.array([-u[1], u[0]])
  fc = (1 - mag, 1 - mag, 1 - mag)
  if sign >= 0:
    hl, hw = 0.050, 0.021
    base = p1 - u * hl
    pts = [p0 + n * stem, base + n * stem, base + n * hw, p1,
           base - n * hw, base - n * stem, p0 - n * stem]
  else:
    bl, bw = 0.016, 0.031
    base = p1 - u * bl
    pts = [p0 + n * stem, base + n * stem, base + n * bw, p1 + n * bw,
           p1 - n * bw, base - n * bw, base - n * stem, p0 - n * stem]
  ax.add_patch(plt.Polygon(pts, closed=True, facecolor=fc,
                           edgecolor='#111111', lw=0.7,
                           joinstyle='miter', zorder=3))


def panel_a(out_dir):
  """The threshold mechanism alone. Input states are black (active) or
  white (inactive); every edge has the same size and its gray level gives
  the weight magnitude, with a bar head for repression. The rule box holds
  a threshold gauge whose needle has just crossed the dashed threshold."""
  fig, axr = plt.subplots(figsize=(6.4, 5.4))
  axr.set_xlim(0, 1.35)
  axr.set_ylim(-0.20, 1.06)
  axr.set_aspect('equal')
  axr.axis('off')

  ins = [(0.96, 1, 0.95), (0.845, 0, 0.30), (0.73, 1, -0.75),
         (0.615, 1, 0.55), (0.50, 0, -0.15), (0.385, 1, 0.10),
         (0.27, 0, -0.65), (0.155, 1, 0.40), (0.04, 0, -0.85)]
  for y, state, w in ins:
    axr.add_patch(Circle((0.075, y), 0.040,
                  facecolor='#1a1a1a' if state else 'white',
                  edgecolor='#000000' if state else '#9aa5b1', lw=1.4))
    y_end = 0.50 + (y - 0.50) * 0.50
    signed_arrow(axr, (0.122, y), (0.455, y_end), abs(w), np.sign(w))

  # the rule box holds a threshold gauge: a semicircle with the flat side
  # down, a dashed line at 90 degrees for the threshold, and a needle just
  # past it, so the target node activates
  axr.add_patch(FancyBboxPatch((0.475, 0.24), 0.38, 0.52,
                boxstyle='round,pad=0.02,rounding_size=0.03',
                facecolor=TILE_BG, edgecolor=INK, lw=1.1))
  cx, cy, R = 0.665, 0.41, 0.175
  from matplotlib.patches import Wedge
  axr.add_patch(Wedge((cx, cy), R, 0, 180, facecolor='white',
                      edgecolor=INK, lw=1.3, zorder=4))
  axr.plot([cx, cx], [cy, cy + R * 0.97], color='#8c8c8c', lw=1.3,
           linestyle=(0, (3.2, 2.6)), zorder=5)
  ang = np.deg2rad(66)
  axr.plot([cx, cx + 0.90 * R * np.cos(ang)], [cy, cy + 0.90 * R * np.sin(ang)],
           color='#c1272d', lw=2.4, zorder=6, solid_capstyle='butt')
  axr.add_patch(Circle((cx, cy), 0.016, facecolor='#c1272d',
                       edgecolor='none', zorder=7))

  signed_arrow(axr, (0.90, 0.50), (1.06, 0.50), 1.0, 1)
  axr.add_patch(Circle((1.13, 0.50), 0.05, facecolor='#1a1a1a',
                       edgecolor='#000000', lw=1.4))
  axr.text(0.03, 1.035, 'inputs at time $t$', fontsize=10.5, color=MUTED,
           ha='left')

  leg_y = -0.115
  axr.add_patch(Circle((0.05, leg_y), 0.020, facecolor='#1a1a1a',
                       edgecolor='#000000', lw=1.0, clip_on=False))
  axr.text(0.082, leg_y, 'active', fontsize=9.5, color=INK, va='center')
  axr.add_patch(Circle((0.245, leg_y), 0.020, facecolor='white',
                       edgecolor='#9aa5b1', lw=1.0, clip_on=False))
  axr.text(0.277, leg_y, 'inactive', fontsize=9.5, color=INK, va='center')
  signed_arrow(axr, (0.47, leg_y), (0.565, leg_y), 0.55, 1)
  axr.text(0.59, leg_y, '$w>0$', fontsize=9.5, color=INK, va='center')
  signed_arrow(axr, (0.72, leg_y), (0.815, leg_y), 0.55, -1)
  axr.text(0.84, leg_y, '$w<0$', fontsize=9.5, color=INK, va='center')

  # colorbar for the magnitude ramp
  cb_x0, cb_x1, cb_y0, cb_y1 = 1.02, 1.30, leg_y - 0.022, leg_y + 0.022
  grad = np.linspace(0, 1, 256).reshape(1, -1)
  axr.imshow(grad, extent=[cb_x0, cb_x1, cb_y0, cb_y1], cmap='gray_r',
             aspect='auto', zorder=2, vmin=0, vmax=1)
  axr.add_patch(plt.Rectangle((cb_x0, cb_y0), cb_x1 - cb_x0, cb_y1 - cb_y0,
                facecolor='none', edgecolor='#111111', lw=0.8, zorder=3))
  axr.text(cb_x0 - 0.015, leg_y, '0', fontsize=9, color=INK, va='center',
           ha='right')
  axr.text(cb_x1 + 0.015, leg_y, '1', fontsize=9, color=INK, va='center',
           ha='left')
  axr.text((cb_x0 + cb_x1) / 2, leg_y + 0.055, '$|w|$', fontsize=9.5,
           color=INK, ha='center', va='bottom')
  save(fig, out_dir, 'fig1b-rule')


# ---------------------------------------------------------------------------
# panel b: Boolean dynamics over time, control vs shocked
# ---------------------------------------------------------------------------
def panel_b(out_dir, trajs, shock_idx=1, t_max=16):
  nodes = pick_display_nodes(trajs, shock_idx=shock_idx, t_max=t_max)
  ctl = trajs[0][nodes, :t_max]
  shk = trajs[shock_idx][nodes, :t_max]
  diff = shk != ctl

  fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.2))
  fig.subplots_adjust(hspace=0.42)
  draw_strip(axes[0], ctl)
  draw_strip(axes[1], shk, diff=diff)
  axes[0].set_title('control network', fontsize=13, color=INK, pad=8)
  axes[1].set_title('shocked network, same initial state', fontsize=13, color=AMBER, pad=8)
  for ax, states in [(axes[0], ctl), (axes[1], shk)]:
    n_rows, n_cols = states.shape
    ax.text(-1.0, n_rows / 2, 'nodes', fontsize=11, color=INK,
            rotation=90, ha='center', va='center')
  n_rows, n_cols = shk.shape
  axes[1].annotate('', xy=(n_cols * 0.35, -1.7), xytext=(0, -1.7),
                   arrowprops=dict(arrowstyle='-|>', lw=1.3, color=INK), annotation_clip=False)
  axes[1].text(n_cols * 0.38, -1.7, 'time', fontsize=11, color=INK, va='center')
  bolt(axes[1], 12.2, n_rows + 1.35, scale=1.15)
  save(fig, out_dir, 'fig1b-dynamics')


# ---------------------------------------------------------------------------
# panel c: one concrete shock, outgoing weights before and after
# ---------------------------------------------------------------------------
def spokes(ax, cx, cy, weights, r=0.24, node_r=0.048, target_state=1):
  ang = np.linspace(0.25 * np.pi, 1.75 * np.pi, len(weights))
  for a, w in zip(ang, weights):
    x2, y2 = cx + r * np.cos(a), cy + r * np.sin(a)
    arrow = FancyArrowPatch(
      (cx, cy), (x2, y2), arrowstyle='-|>', mutation_scale=10,
      shrinkA=12, shrinkB=6,
      lw=0.7 + 2.6 * abs(w), color=edge_color(w), capstyle='round', zorder=2,
    )
    ax.add_patch(arrow)
    ax.add_patch(Circle((x2, y2), 0.032, facecolor='white', edgecolor='#9aa5b1', lw=1.1, zorder=3))
    lx, ly = cx + (r + 0.085) * np.cos(a), cy + (r + 0.085) * np.sin(a)
    ax.text(lx, ly, f'${w:+.1f}$'.replace('+1.0', '+1').replace('-1.0', '-1'),
            fontsize=9.5, ha='center', va='center', color=MUTED)
  ax.add_patch(Circle((cx, cy), node_r,
               facecolor=BLUE if target_state else 'white',
               edgecolor=AMBER, lw=2.4, zorder=5))


def panel_c(out_dir):
  fig, ax = plt.subplots(figsize=(8.0, 3.4))
  ax.set_xlim(0, 2.35)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal')
  ax.axis('off')

  before = [0.8, -0.3, 0.5, -0.7]
  after = [-1.0, 1.0, 1.0, -1.0]

  spokes(ax, 0.42, 0.48, before)
  spokes(ax, 1.93, 0.48, after)
  ax.text(0.42, 0.055, 'before', fontsize=12, ha='center', color=INK)
  ax.text(1.93, 0.055, 'after', fontsize=12, ha='center', color=AMBER)

  bolt(ax, 1.175, 0.76, scale=1.2)
  arrow = FancyArrowPatch((0.86, 0.48), (1.48, 0.48), arrowstyle='-|>',
                          mutation_scale=15, lw=1.8, color=AMBER)
  ax.add_patch(arrow)
  ax.text(1.175, 0.585, 'shock', fontsize=12.5, color=AMBER, ha='center')
  ax.text(1.175, 0.345, "$w' = \\pm 1$", fontsize=11.5, color=INK, ha='center')
  ax.text(1.175, 0.235, 'redrawn at random', fontsize=10.5, color=MUTED, ha='center')

  ax.text(0.42, 0.93, 'outgoing weights of one target node', fontsize=11.5, color=INK, ha='center')
  save(fig, out_dir, 'fig1c-shock-closeup')


# ---------------------------------------------------------------------------
# panel d: the identification problem
# ---------------------------------------------------------------------------
def panel_d(out_dir, trajs, m=5, t0=28, t_max=40, true_shock=2, seed=12):
  rng = np.random.default_rng(seed)
  reporters = rng.choice(trajs[0].shape[0], size=m, replace=False)
  obs = trajs[true_shock][reporters, t0:t_max]

  fig, ax = plt.subplots(figsize=(9.6, 3.0))
  ax.set_xlim(0, 3.4)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal')
  ax.axis('off')

  cell = 0.09
  x0, y0 = 0.12, 0.30
  n_rows, n_cols = obs.shape
  for r in range(n_rows):
    for c in range(n_cols):
      on = obs[r, c] == 1
      ax.add_patch(Rectangle((x0 + c * cell, y0 + (n_rows - 1 - r) * cell), cell, cell,
                   facecolor=BLUE if on else OFF, edgecolor='white', linewidth=1.0, zorder=2))
  ax.text(x0 + n_cols * cell / 2, 0.86, f'$m={m}$ reporters observed',
          fontsize=12, ha='center', color=INK)
  ax.text(x0 + n_cols * cell / 2, 0.175, 'one noisy trial, late times',
          fontsize=10.5, ha='center', color=MUTED)

  bx = x0 + n_cols * cell + 0.16
  arrow = FancyArrowPatch((bx, 0.53), (bx + 0.22, 0.53), arrowstyle='-|>',
                          mutation_scale=13, lw=1.6, color=INK)
  ax.add_patch(arrow)
  ax.add_patch(FancyBboxPatch((bx + 0.26, 0.38), 0.62, 0.30,
               boxstyle='round,pad=0.02,rounding_size=0.04',
               facecolor=TILE_BG, edgecolor=INK, lw=1.2))
  ax.text(bx + 0.57, 0.565, 'classifier', fontsize=12.5, ha='center', color=INK)
  ax.text(bx + 0.57, 0.45, 'which shock?', fontsize=10.5, ha='center', color=MUTED)
  arrow = FancyArrowPatch((bx + 0.92, 0.53), (bx + 1.14, 0.53), arrowstyle='-|>',
                          mutation_scale=13, lw=1.6, color=INK)
  ax.add_patch(arrow)

  options = ['control', 'shock 1', 'shock 2', 'shock 3']
  ys = [0.82, 0.62, 0.42, 0.22]
  for lab, y in zip(options, ys):
    chosen = lab == f'shock {true_shock}'
    ax.text(
      bx + 1.20, y, lab, fontsize=11.5,
      color='white' if chosen else MUTED,
      fontweight='bold' if chosen else 'normal',
      bbox=dict(
        boxstyle='round,pad=0.30',
        facecolor=AMBER if chosen else '#f0f2f5',
        edgecolor=AMBER if chosen else '#d5d9df',
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
  panel_b(args.out_dir, trajs)
  panel_c(args.out_dir)
  panel_d(args.out_dir, trajs)


if __name__ == '__main__':
  main()
