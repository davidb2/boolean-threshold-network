#!/usr/bin/env python3
'''Compose the two halves of main text Figure 3 into a single image.

The top half is the 2x2 panel size grid (a to d) and the bottom half the
class tree beside the example panel (e and f). The bottom image is scaled
to the top image's width and the two are stacked on a white canvas with a
small gap, so figures.tex includes exactly one file for the figure.

Usage:
  python scripts/compose-fig3.py \
    --top pics/fig-panel-size/fig-panel-size.png \
    --bottom pics/fig-disks/fig-disks-eps1-row.png \
    --out pics/fig3/fig3.png
'''
import argparse
import pathlib

from PIL import Image


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--top', required=True)
  p.add_argument('--bottom', required=True)
  p.add_argument('--gap', type=int, default=70)
  p.add_argument('--out', required=True)
  args = p.parse_args()

  top = Image.open(args.top).convert('RGB')
  bot = Image.open(args.bottom).convert('RGB')
  w = top.width
  bot = bot.resize((w, round(bot.height * w / bot.width)), Image.LANCZOS)
  canvas = Image.new('RGB', (w, top.height + args.gap + bot.height), 'white')
  canvas.paste(top, (0, 0))
  canvas.paste(bot, (0, top.height + args.gap))
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(out, dpi=(300, 300))
  print(f'wrote {out} ({canvas.width}x{canvas.height})')


if __name__ == '__main__':
  main()
