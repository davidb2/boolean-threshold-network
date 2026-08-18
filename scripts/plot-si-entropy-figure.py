#!/usr/bin/env python3
'''Classifying reporters by response breadth rather than by mean response.

The scalar sensitivity S averages a node's response over the ten shocks,
so it cannot separate a node that answers every shock weakly from a node
that answers one shock enormously and the rest not at all. This figure
splits nodes on two measured axes instead:

  amplitude  max_q Delta_{j,q}, the largest single shock response
  breadth    H_j = -sum_q p log p / log 10, the entropy of the node's
             normalized response profile p_q = Delta_{j,q} / sum Delta,
             so H = 1 means it answers all ten shocks equally and H -> 0
             means its whole response sits in one shock

  unresponsive   amplitude below 0.05
  indiscriminate responsive with H >= 0.8
  dormant        responsive with H < 0.8

  a  the two axes together. Broad responders pile against the
     decorrelation ceiling near amplitude 0.5 at H close to one, while
     the largest single responses belong to selective nodes.
  b  what each selection strategy puts in an eight member panel. The
     comparison that matters is against the responsiveness heuristic, the
     rule the paper argues with, not against random selection.
  c  the same composition at three noise levels.
  d  breadth is not a restatement of amplitude: the relation is not
     monotone, and the most extreme responders are selective, not broad.

Usage:
  python scripts/plot-si-entropy-figure.py \
    --sensitivity-dir data/sensitivity \
    --ga-csvs <eps0> <eps0.5> <eps1> --tags 1.0 0.75-b4 0.5 \
    --out-dir plots/si-entropy
'''
import argparse
import ast
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IND = '#ff7f0e'      # indiscriminate
DOR = '#000000'      # dormant
UNR = '#c7c7c7'      # unresponsive
AMP_CUT = 0.05
H_CUT = 0.80
K = 8

plt.rcParams.update({
  'font.size': 18,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def profile_stats(sens_dir, tag):
  '''Amplitude and normalized profile entropy for every node.'''
  Sp = np.load(f'{sens_dir}/S-perdrug-rho{tag}.npz')['S']     # (nets, shocks, nodes)
  tot = Sp.sum(axis=1)
  amp = Sp.max(axis=1)
  p = np.divide(Sp, tot[:, None, :], out=np.zeros_like(Sp), where=tot[:, None, :] > 0)
  with np.errstate(divide='ignore', invalid='ignore'):
    H = -(np.where(p > 0, p * np.log(p), 0)).sum(axis=1) / np.log(Sp.shape[1])
  H[tot == 0] = np.nan
  return amp, H


def classes(amp_row, H_row, nodes):
  out = []
  for n in nodes:
    if amp_row[n] < AMP_CUT or np.isnan(H_row[n]):
      out.append('unresponsive')
    elif H_row[n] >= H_CUT:
      out.append('indiscriminate')
    else:
      out.append('dormant')
  return out


def panels_for(ga_csv):
  ga = pd.read_csv(ga_csv)
  ga = ga[ga.max_num_features == K]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  out = {}
  for _, r in fin.iterrows():
    nodes = [int(s.split('-')[1]) for s in ast.literal_eval(r['features'])]
    if len(set(nodes)) == K:
      out[int(r['original_network_idx'])] = nodes
  return out


def composition(sens_dir, tag, ga_csv, rng):
  '''Mean class counts per panel for evolved, heuristic, and random selection.'''
  amp, H = profile_stats(sens_dir, tag)
  b = np.load(f'{sens_dir}/B-rho{tag}.npz')
  S, bnets = b['B'], [int(x) for x in b['networks']]
  panels = panels_for(ga_csv)
  rows = []
  for net, nodes in panels.items():
    bi = bnets.index(net)
    top = list(np.argsort(-S[bi])[:K])
    rnd = list(rng.choice(S.shape[1], K, replace=False))
    for lab, sel in [('evolved', nodes), ('most responsive', top), ('random', rnd)]:
      for c in classes(amp[bi], H[bi], sel):
        rows.append(dict(strategy=lab, cls=c))
  T = pd.DataFrame(rows)
  n = len(panels)
  return (T.groupby(['strategy', 'cls']).size().unstack(fill_value=0) / n
          ).reindex(index=['evolved', 'most responsive', 'random'],
                    columns=['indiscriminate', 'dormant', 'unresponsive'], fill_value=0)


def stacked(ax, comp, title=None, legend=False):
  ys = np.arange(len(comp))
  left = np.zeros(len(comp))
  for cls, col in [('indiscriminate', IND), ('dormant', DOR), ('unresponsive', UNR)]:
    v = comp[cls].to_numpy()
    ax.barh(ys, v, left=left, color=col, height=0.62,
            edgecolor='white', linewidth=1.2, label=cls if legend else None)
    for y, (l, w) in enumerate(zip(left, v)):
      if w > 0.55:
        ax.text(l + w / 2, y, f'{w:.1f}', ha='center', va='center', fontsize=13,
                color='white' if cls != 'unresponsive' else '#444444')
    left += v
  ax.set_yticks(ys)
  ax.set_yticklabels(comp.index)
  ax.set_xlim(0, K)
  ax.set_xlabel('Members per panel')
  ax.invert_yaxis()
  if title:
    ax.set_title(title, fontsize=17)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', required=True)
  p.add_argument('--ga-csvs', nargs=3, required=True)
  p.add_argument('--tags', nargs=3, required=True)      # rho tags for eps 0, 0.5, 1
  p.add_argument('--eps-labels', nargs=3, default=['0', '0.5', '1'])
  p.add_argument('--out-dir', required=True)
  args = p.parse_args()
  rng = np.random.default_rng(5)

  fig = plt.figure(figsize=(15.2, 9.6))
  gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28)
  ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
  ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

  # a: the two axes, at high noise
  amp, H = profile_stats(args.sensitivity_dir, args.tags[2])
  m = (amp >= AMP_CUT) & ~np.isnan(H)
  ax_a.hist2d(amp[m], H[m], bins=[np.linspace(0, 1, 70), np.linspace(0, 1.001, 70)],
              cmap='Greys', norm=matplotlib.colors.LogNorm())
  ax_a.axhline(H_CUT, color=IND, lw=1.8, linestyle=(0, (4, 3)))
  ax_a.text(0.02, H_CUT + 0.02, 'indiscriminate', fontsize=14, color=IND)
  ax_a.text(0.02, H_CUT - 0.07, 'dormant', fontsize=14, color=DOR)
  ax_a.set_xlabel('Amplitude, $\\max_q \\Delta_{j,q}$')
  ax_a.set_ylabel('Breadth, profile entropy $H$')
  ax_a.set_title(f'Responsive nodes, $\\varepsilon = {args.eps_labels[2]}$', fontsize=17)

  # b: composition by strategy at high noise -- the headline panel
  comp_hi = composition(args.sensitivity_dir, args.tags[2], args.ga_csvs[2], rng)
  stacked(ax_b, comp_hi, title=f'$\\varepsilon = {args.eps_labels[2]}$', legend=True)
  ax_b.legend(frameon=False, fontsize=13, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), columnspacing=1.2, handlelength=1.1)
  print(comp_hi.round(2).to_string())

  # c: the same at all three noise levels, evolved vs the heuristic
  width = 0.26
  xs = np.arange(3)
  for j, (lab, col) in enumerate([('evolved', DOR), ('most responsive', IND),
                                  ('random', UNR)]):
    vals = []
    for tag, ga in zip(args.tags, args.ga_csvs):
      c = composition(args.sensitivity_dir, tag, ga, np.random.default_rng(5))
      vals.append(c.loc[lab, 'dormant'] + c.loc[lab, 'unresponsive'])
    ax_c.bar(xs + (j - 1) * width, vals, width * 0.92, color=col,
             edgecolor='#222222', linewidth=1.0, label=lab)
    for x, v in zip(xs + (j - 1) * width, vals):
      ax_c.text(x, v + 0.06, f'{v:.2f}', ha='center', fontsize=12)
  ax_c.set_xticks(xs)
  ax_c.set_xticklabels([f'$\\varepsilon = {e}$' for e in args.eps_labels])
  ax_c.set_ylabel('Non indiscriminate\nmembers per panel')
  ax_c.set_ylim(0, 5.2)
  ax_c.legend(frameon=False, fontsize=12.5, loc='upper right', ncol=1,
              handlelength=1.1, labelspacing=0.3)

  # d: breadth is not amplitude
  qs = np.quantile(amp[m], np.linspace(0, 1, 9))
  mids, meds, fsel = [], [], []
  for i in range(len(qs) - 1):
    sel = (amp[m] >= qs[i]) & (amp[m] <= qs[i + 1])
    if sel.sum() < 50:
      continue
    mids.append(0.5 * (qs[i] + qs[i + 1]))
    meds.append(np.nanmedian(H[m][sel]))
    fsel.append(np.nanmean(H[m][sel] < H_CUT))
  ax_d.plot(mids, meds, color='#222222', lw=2.2, marker='o', markersize=5)
  ax_d.set_xlabel('Amplitude, $\\max_q \\Delta_{j,q}$')
  ax_d.set_ylabel('Median profile entropy $H$')
  ax_d.axhline(H_CUT, color=IND, lw=1.4, linestyle=(0, (4, 3)))
  ax_d.set_ylim(0, 1.05)
  ax2 = ax_d.twinx()
  ax2.plot(mids, fsel, color=DOR, lw=1.6, linestyle=(0, (3, 2)), marker='s', markersize=4)
  ax2.set_ylabel('Fraction dormant', fontsize=15)
  ax2.set_ylim(0, 1.0)
  ax2.spines['top'].set_visible(False)
  ax_d.text(0.42, 0.30, 'decorrelation\nceiling', fontsize=13, color='#666666', ha='center')

  for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], 'abcd'):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'si-entropy.svg', bbox_inches='tight')
  fig.savefig(out / 'si-entropy.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out}/si-entropy.svg + .png')


if __name__ == '__main__':
  main()
