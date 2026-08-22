#!/usr/bin/env python3
"""Compose Figure 1 into one image.

Row 1: a (application domains, BioRender) on the left, with b (the
threshold rule) over c (one shock in detail) on the right, and the weight
colorbar standing between b and c.
Row 2: d (control and 2 shocked copies) on the left, with e (what noise
does to the initial condition) over f (the inference task) on the right.
Each right column is sized so that its 2 panels, plus the gap between
them, match the height of the panel beside them. Panels are cropped to
their content, and each letter sits in the free corner of its own panel,
or just outside it where the corner is taken, so no vertical space has to
be reserved for the letters at all.

Usage:
  python scripts/compose-fig1.py --pics-dir <paper>/pics/fig1
"""
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INK = '#222222'

FILES = [('a', 'fig1a-domains.png'), ('b', 'fig1b-rule.png'),
         ('c', 'fig1c-shock-closeup.png'), ('d', 'fig1d-dynamics.png'),
         ('e', 'fig1e-noise.png'), ('f', 'fig1f-inference.png')]

# Where each panel letter goes, as an offset in inches from the top left
# corner of the panel. Panels whose corner is free carry the letter
# inside; the panels whose top row starts with a node or a title carry it
# just outside, in the gutter, so nothing has to be reserved above.
LETTERS = {'a': ('in', 0.03, 0.02), 'b': ('out', 0.40, 0.00),
           'c': ('out', 0.40, 0.10), 'd': ('in', 0.00, 0.05),
           'e': ('out', 0.40, 0.00), 'f': ('out', 0.40, 0.05)}


def trim(im):
  """Crop a panel to its visible content, so panels butt up against each
  other and the letters land on the drawing rather than on an exported
  margin."""
  mask = im.convert('L').point(lambda v: 255 if v < 250 else 0)
  box = mask.getbbox()
  return im.crop(box) if box else im


def column_width(ar_left, ar_top, ar_bot, span_in, band):
  """Width in inches of a right column of 2 stacked panels whose total
  height, including the gap between them, matches the height of the panel
  of aspect ar_left that fills the rest of the span."""
  return ((span_in / ar_left - band)
          / (1 / ar_top + 1 / ar_bot + 1 / ar_left))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--pics-dir', required=True)
  args = p.parse_args()
  d = pathlib.Path(args.pics_dir)

  ims = {k: trim(Image.open(d / f)) for k, f in FILES}
  ar = {k: im.size[0] / im.size[1] for k, im in ims.items()}

  W = 12.0                      # figure width in inches
  M, GX, GY = 0.03, 0.055, 0.030   # margins and gutters, figure fraction
  CB = 0.95                     # inches reserved for the shared colorbar

  gy = GY * W
  band = gy                     # gap between the 2 panels of a column
  span1 = 1 - 2 * M - GX - CB / W
  span2 = 1 - 2 * M - GX

  u1 = column_width(ar['a'], ar['b'], ar['c'], span1 * W, band)
  wr1, wa = u1 / W, span1 - u1 / W
  ha, hb, hc = wa * W / ar['a'], u1 / ar['b'], u1 / ar['c']

  u2 = column_width(ar['d'], ar['e'], ar['f'], span2 * W, band)
  wr2, wd = u2 / W, span2 - u2 / W
  hd, he, hf = wd * W / ar['d'], u2 / ar['e'], u2 / ar['f']

  H = ha + hd + gy + 0.14
  fig = plt.figure(figsize=(W, H))

  def put(key, x, y_top_in, w_frac, h_in):
    h_frac = h_in / H
    ax = fig.add_axes([x, 1 - (y_top_in + h_in) / H, w_frac, h_frac])
    ax.imshow(ims[key])
    ax.axis('off')
    mode, dx, dy = LETTERS[key]
    lx = x + (dx if mode == 'in' else -dx) / W
    fig.text(lx, 1 - (y_top_in + dy) / H, key, fontsize=26,
             fontweight='bold', family='sans-serif', color=INK,
             ha='left', va='top')

  # panels in a row hang from the same top line, so their letters align
  y1 = 0.08
  put('a', M, y1, wa, ha)
  put('b', M + wa + GX, y1, wr1, hb)
  put('c', M + wa + GX, y1 + hb + band, wr1, hc)

  # one colorbar for the weight shading in b and c, standing off to the
  # right of b and centered on its column of nodes
  cbh, cbw = 1.9, 0.20 / W
  cbx = M + wa + GX + wr1 + 0.30 / W
  cby = y1 + hb * 0.48
  cax = fig.add_axes([cbx, 1 - (cby + cbh / 2) / H, cbw, cbh / H])
  cax.imshow(np.linspace(1, 0, 256).reshape(-1, 1), cmap='gray_r',
             aspect='auto', vmin=0, vmax=1)
  cax.set_xticks([])
  cax.set_yticks([])
  for sp in cax.spines.values():
    sp.set_linewidth(0.8)
    sp.set_color('#111111')
  for lab, yy in [('1', cby - cbh / 2), ('0', cby + cbh / 2)]:
    fig.text(cbx + cbw + 0.05 / W, 1 - yy / H, lab, fontsize=11, color=INK,
             ha='left', va='center', family='serif')
  fig.text(cbx + cbw + 0.42 / W, 1 - cby / H, 'Connection strength',
           fontsize=11.5, color=INK, ha='center', va='center', rotation=90,
           family='serif')

  y2 = y1 + ha + gy
  put('d', M, y2, wd, hd)
  put('e', M + wd + GX, y2, wr2, he)
  put('f', M + wd + GX, y2 + he + band, wr2, hf)

  fig.savefig(d / 'fig1.png', dpi=600)
  fig.savefig(d / 'fig1.svg')
  print(f'wrote {d}/fig1.png + .svg ({W:.1f}x{H:.1f} in)')


if __name__ == '__main__':
  main()
