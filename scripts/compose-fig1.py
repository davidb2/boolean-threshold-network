#!/usr/bin/env python3
"""Compose Figure 1 into one image.

Row 1: a (application domains, BioRender) beside b (the threshold rule).
Row 2: c (one shock in detail), full width.
Row 3: d (control vs shocked dynamics) beside e (the inference task).
Panel letters sit above each panel, so no panel content rises past its
letter.

Usage:
  python scripts/compose-fig1.py --pics-dir <paper>/pics/fig1
"""
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

INK = '#222222'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--pics-dir', required=True)
  args = p.parse_args()
  d = pathlib.Path(args.pics_dir)

  ims = {k: Image.open(d / f) for k, f in [
      ('a', 'fig1a-domains.png'), ('b', 'fig1b-rule.png'),
      ('c', 'fig1c-shock-closeup.png'), ('d', 'fig1b-dynamics.png'),
      ('e', 'fig1d-inference.png')]}
  ar = {k: im.size[0] / im.size[1] for k, im in ims.items()}

  W = 12.0                      # figure width in inches
  M, GX, GY = 0.03, 0.035, 0.030   # margins and gutters, figure fraction
  LETTER_BAND = 0.24            # inches reserved above each panel

  # row 1: a gets 56 percent of the span, b the rest
  span = 1 - 2 * M - GX
  wa, wb = span * 0.56, span * 0.44
  ha = wa * W / ar['a']
  hb = wb * W / ar['b']
  h1 = max(ha, hb)
  # row 2: c across the full span
  wc = 1 - 2 * M
  h2 = wc * W / ar['c']
  # row 3: d sized by its share, e fills the rest
  wd = span * 0.32
  we = span * 0.68
  h3 = max(wd * W / ar['d'], we * W / ar['e'])

  H = h1 + h2 + h3 + 3 * LETTER_BAND + 2 * GY * W + 0.45
  fig = plt.figure(figsize=(W, H))

  def put(key, x, y_top_in, w_frac, h_in, letter_x=None):
    im = ims[key]
    h_frac = h_in / H
    y = 1 - (y_top_in + h_in) / H
    ax = fig.add_axes([x, y, w_frac, h_frac])
    ax.imshow(im)
    ax.axis('off')
    lx = letter_x if letter_x is not None else x
    fig.text(lx, 1 - (y_top_in - 0.06) / H, key, fontsize=26,
             fontweight='bold', family='sans-serif', color=INK,
             ha='left', va='bottom')

  y0 = LETTER_BAND + 0.30
  put('a', M, y0 + (h1 - ha) / 2, wa, ha)
  put('b', M + wa + GX, y0 + (h1 - hb) / 2, wb, hb)
  y0 += h1 + GY * W + LETTER_BAND
  put('c', M, y0, wc, h2)
  y0 += h2 + GY * W + LETTER_BAND
  put('d', M, y0 + (h3 - wd * W / ar['d']) / 2, wd, wd * W / ar['d'])
  put('e', M + wd + GX, y0 + (h3 - we * W / ar['e']) / 2, we,
      we * W / ar['e'])

  fig.savefig(d / 'fig1.png', dpi=300)
  fig.savefig(d / 'fig1.svg')
  print(f'wrote {d}/fig1.png + .svg ({W:.1f}x{H:.1f} in)')


if __name__ == '__main__':
  main()
