#!/usr/bin/env python3
"""Compose Figure 1 into one image.

Row 1: a (application domains, BioRender) on the left, with b (the
threshold rule) over c (one shock in detail) on the right, sized so the
right column matches the height of a.
Row 2: d (control vs shocked dynamics) beside e (the inference task).
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

  span = 1 - 2 * M - GX
  gy = GY * W
  # row 1: a on the left, b stacked over c on the right. The column width
  # is set so that b, its letter band, and c together match the height of
  # a, which fixes every size in the row.
  band = LETTER_BAND + gy
  u = (span * W / ar['a'] - band) / (1 / ar['b'] + 1 / ar['c'] + 1 / ar['a'])
  wr = u / W
  wa = span - wr
  ha = wa * W / ar['a']
  hb, hc = u / ar['b'], u / ar['c']
  h1 = ha
  # row 2: d kept narrow because it is a tall raster, e fills the rest
  wd, we = span * 0.32, span * 0.68
  hd = wd * W / ar['d']
  he = we * W / ar['e']
  h2 = max(hd, he)

  H = h1 + h2 + 2 * LETTER_BAND + gy + 0.45
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

  # panels in a row hang from the same top line, so their letters align
  y0 = LETTER_BAND + 0.30
  put('a', M, y0, wa, ha)
  put('b', M + wa + GX, y0, wr, hb)
  put('c', M + wa + GX, y0 + hb + band, wr, hc)
  y0 += h1 + gy + LETTER_BAND
  put('d', M, y0, wd, hd)
  put('e', M + wd + GX, y0, we, he)

  fig.savefig(d / 'fig1.png', dpi=300)
  fig.savefig(d / 'fig1.svg')
  print(f'wrote {d}/fig1.png + .svg ({W:.1f}x{H:.1f} in)')


if __name__ == '__main__':
  main()
