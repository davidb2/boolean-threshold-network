#!/usr/bin/env python3
'''How monitored sets are chosen in practice (SI figure, replaces Table S1).

A left to right merging tree. 18 studies, colored by domain, flow into the
criterion that fixed their analyzed set, and the criteria flow into a two node
verdict: ranked by expected discrimination or not. 17 of the 18 converge on
"not ranked"; the one exception (Lal et al., EEG channel selection) ranks
sensors rather than biological components, and its path is drawn bold.

Several pipelines apply a coarse pre-filter first (Zaslaver's intergenic rule,
Bollenbach's curated gene list, Quian Quiroga's clinical placement, Chen's
brightness threshold); the node shown is the LAST filter, the one that fixed
the analyzed set. The SI prose keeps the full pipelines, and the caption
states the rule.

No external data. Render locally:
  python scripts/plot-si-practice-figure.py --out-dir plots/si
'''
import argparse
import pathlib
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
  'font.size': 15,
  'mathtext.fontset': 'cm',
  'svg.fonttype': 'none',
})

# domain -> color (colorblind safe, distinct from the class colors of the
# other figures so no accidental promiscuous/dormant reading)
DOMAINS = {
  'bacterial gene expression': '#0072B2',
  'neuroscience':              '#E69F00',
  'human behavior':            '#009E73',
  'urban sensing':             '#B2559D',
  'sensor engineering':        '#CC3311',
}

# (study label, domain, criterion) in figure order, top to bottom.
# The criterion is the last filter that fixed the analyzed set.
STUDIES = [
  ('Zaslaver et al. 2006',      'bacterial gene expression', 'detect'),
  ('Bollenbach et al. 2009',    'bacterial gene expression', 'detect'),
  ('Mitosch et al. 2019',       'bacterial gene expression', 'resp'),
  ('Mohiuddin et al. 2022',     'bacterial gene expression', 'resp'),
  ('Dash et al. 2024',          'bacterial gene expression', 'topo'),
  ('Quian Quiroga et al. 2005', 'neuroscience',              'resp'),
  ('Chen et al. 2013',          'neuroscience',              'resp'),
  ('Kanwisher et al. 1997',     'neuroscience',              'resp'),
  ('Rust & DiCarlo 2010',       'neuroscience',              'all'),
  ('Marre et al. 2012',         'neuroscience',              'all'),
  ('Schneidman et al. 2006',    'neuroscience',              'all'),
  ('de Vries et al. 2020',      'neuroscience',              'all'),
  ('Steinmetz et al. 2019',     'neuroscience',              'anat'),
  ('Saeb et al. 2015',          'human behavior',            'prior'),
  ('Mishra et al. 2020',        'human behavior',            'prior'),
  ('Schmidt et al. 2018',       'human behavior',            'avail'),
  ("O'Keeffe et al. 2019",      'urban sensing',             'avail'),
  ('Lal et al. 2004',           'sensor engineering',        'class'),
]

# criterion nodes in vertical order, top to bottom
CRITERIA = [
  ('detect', 'detectability\nthreshold'),
  ('resp',   'responsiveness to the\nstimulus or perturbation'),
  ('topo',   'structural or\ntopological rule'),
  ('all',    'everything the\ninstrument isolates'),
  ('anat',   'anatomical\ntargeting'),
  ('prior',  'prior knowledge or\nclinical reasoning'),
  ('avail',  'platform availability,\nopportunistic'),
  ('class',  'classification\nperformance'),
]

X_STUDY, X_CRIT, X_VERDICT = 0.115, 0.475, 0.855
W_STUDY, W_CRIT, W_VERDICT = 0.21, 0.20, 0.235
GRAY_FACE, GRAY_EDGE, INK = '#f2f2f2', '#8a8a8a', '#222222'


def box(ax, x, y, w, h, face, edge, lw=1.4):
  ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle='round,pad=0.004,rounding_size=0.011',
                              facecolor=face, edgecolor=edge, lw=lw, zorder=4))


def edge_curve(ax, x0, y0, x1, y1, color, lw=1.5, alpha=0.85, rad=0.10, arrow=False):
  style = '-|>' if arrow else '-'
  ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=2,
              arrowprops=dict(arrowstyle=style, color=color, lw=lw, alpha=alpha,
                              shrinkA=0, shrinkB=0,
                              connectionstyle=f'arc3,rad={rad}',
                              mutation_scale=13))


def lighten(hex_color, f=0.86):
  r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
  mix = lambda c: int(c + (255 - c) * f)
  return f'#{mix(r):02x}{mix(g):02x}{mix(b):02x}'


def main(out_dir):
  fig, ax = plt.subplots(figsize=(13.6, 11.0))
  ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.01); ax.axis('off')

  # --- study rows: 18 rows with a gap between domain blocks
  y_top, y_bot, gap = 0.895, 0.075, 1.15   # gap in row units
  blocks = []
  prev = None
  n_gaps = 0
  for _, dom, _ in STUDIES:
    if dom != prev:
      n_gaps += 1 if prev is not None else 0
      prev = dom
  step = (y_top - y_bot) / (len(STUDIES) - 1 + gap * n_gaps)
  ys, y, prev = [], y_top, None
  for _, dom, _ in STUDIES:
    if prev is not None and dom != prev:
      y -= gap * step
    ys.append(y)
    y -= step
    prev = dom
  row_h = step * 0.78

  for (label, dom, crit), yy in zip(STUDIES, ys):
    c = DOMAINS[dom]
    bold = crit == 'class'
    box(ax, X_STUDY, yy, W_STUDY, row_h, lighten(c), c, lw=2.6 if bold else 1.4)
    ax.text(X_STUDY, yy, label, ha='center', va='center', fontsize=11.6,
            color=INK, zorder=6,
            fontweight='bold' if bold else 'normal')

  # domain names as small horizontal captions above each block
  import itertools
  for dom, grp in itertools.groupby(zip(STUDIES, ys), key=lambda t: t[0][1]):
    grp = list(grp)
    ax.text(X_STUDY - W_STUDY / 2, grp[0][1] + row_h / 2 + 0.012,
            dom, ha='left', va='bottom', fontsize=10.8,
            color=DOMAINS[dom], fontweight='bold')

  # --- criterion nodes at the mean height of their feeders, min spacing apart
  feeders = {k: [yy for (_, _, cr), yy in zip(STUDIES, ys) if cr == k]
             for k, _ in CRITERIA}
  crit_y = {}
  min_gap = 0.103
  last = 1.1
  for k, _ in CRITERIA:
    target = sum(feeders[k]) / len(feeders[k])
    yy = max(min(target, last - min_gap), 0.045)
    crit_y[k] = yy
    last = yy
  crit_h = 0.075

  for k, label in CRITERIA:
    bold = k == 'class'
    box(ax, X_CRIT, crit_y[k], W_CRIT, crit_h, GRAY_FACE,
        INK if bold else GRAY_EDGE, lw=2.6 if bold else 1.4)
    n = len(feeders[k])
    ax.text(X_CRIT, crit_y[k], label, ha='center', va='center',
            fontsize=11.2, color=INK, zorder=6,
            fontweight='bold' if bold else 'normal')
    ax.text(X_CRIT + W_CRIT / 2 - 0.008, crit_y[k] + crit_h / 2 - 0.008,
            f'{n}', ha='right', va='top', fontsize=9.5, color='#666666',
            zorder=7, style='italic')

  # --- study -> criterion edges, domain colored
  for (label, dom, crit), yy in zip(STUDIES, ys):
    bold = crit == 'class'
    edge_curve(ax, X_STUDY + W_STUDY / 2 + 0.004, yy,
               X_CRIT - W_CRIT / 2 - 0.004, crit_y[crit],
               DOMAINS[dom], lw=3.0 if bold else 1.5,
               rad=0.10 if yy >= crit_y[crit] else -0.10)

  # --- verdict nodes
  not_ranked_feeders = [k for k, _ in CRITERIA if k != 'class']
  y_not = sum(crit_y[k] for k in not_ranked_feeders) / len(not_ranked_feeders)
  y_yes = crit_y['class']
  v_h = 0.10
  box(ax, X_VERDICT, y_not, W_VERDICT, v_h, GRAY_FACE, GRAY_EDGE, lw=1.4)
  ax.text(X_VERDICT, y_not + 0.012, 'not ranked by expected\ndiscrimination',
          ha='center', va='center', fontsize=12.2, color=INK, zorder=6)
  ax.text(X_VERDICT, y_not - 0.031, '17 studies', ha='center', va='center',
          fontsize=10.5, color='#666666', style='italic', zorder=6)

  box(ax, X_VERDICT, y_yes, W_VERDICT, v_h, '#fdeeea', '#CC3311', lw=2.6)
  ax.text(X_VERDICT, y_yes + 0.012, 'ranked by expected\ndiscrimination',
          ha='center', va='center', fontsize=12.2, color=INK,
          fontweight='bold', zorder=6)
  ax.text(X_VERDICT, y_yes - 0.030, '1 study',
          ha='center', va='center', fontsize=10.5, color='#8c2010',
          style='italic', zorder=6)

  # --- criterion -> verdict edges
  for k, _ in CRITERIA:
    bold = k == 'class'
    ytgt = y_yes if bold else y_not
    edge_curve(ax, X_CRIT + W_CRIT / 2 + 0.004, crit_y[k],
               X_VERDICT - W_VERDICT / 2 - 0.006, ytgt,
               '#CC3311' if bold else GRAY_EDGE,
               lw=3.0 if bold else 1.6, alpha=0.9,
               rad=0.10 if crit_y[k] >= ytgt else -0.10, arrow=True)

  # --- column headers
  hdr_y = 0.985
  for x, t in [(X_STUDY, 'Study'), (X_CRIT, 'Monitored set fixed by'),
               (X_VERDICT, 'Components ranked by\ndiscrimination?')]:
    ax.text(x, hdr_y, t, ha='center', va='top', fontsize=13.5,
            color=INK, fontweight='bold')

  out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
  for ext, kw in [('png', dict(dpi=600)), ('svg', dict())]:
    fig.savefig(out / f'si-practice.{ext}', bbox_inches='tight', **kw)
  plt.close(fig)
  print(f'wrote {out}/si-practice.svg + .png')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', type=str, default='plots/si')
  main(p.parse_args().out_dir)
