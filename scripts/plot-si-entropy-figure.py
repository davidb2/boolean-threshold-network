#!/usr/bin/env python3
'''Classifying reporters by response breadth rather than by mean response.

The scalar sensitivity S averages a node's response over the ten shocks,
so a node that answers one shock enormously and a node that answers every
shock weakly can carry the same S. This figure classifies nodes by how
many shocks they answer in absolute terms instead.

A node answers shock q when its deviation reaches the sensitivity cutoff,
Delta_{j,q} >= theta, where theta is the antimode of the pooled S
distribution, the same cutoff that separates the sensitivity classes. Let
n_j be the number of shocks a node answers. Then

  unresponsive    n_j = 0    no shock ever moves it past the cutoff
  dormant         1 <= n_j <= 5   it answers a minority of shocks
  promiscuous  n_j >= 6   it answers a majority of shocks

The rule introduces no threshold beyond theta, and it enforces the
requirement that a dormant node must respond strongly to something.

  a  the distribution of n_j at three noise levels. It is U shaped with an
     interior minimum near the majority split, so the split is not
     arbitrary.
  b  mean sensitivity against breadth. S predicts n_j well but not
     exactly, and the disagreement sits at the cutoff, which is where the
     mean is least informative.
  c  what each selection strategy puts in an eight member panel. The
     comparison that matters is against the responsiveness heuristic, the
     rule the paper argues with, not against random selection.
  d  the same composition at three noise levels.

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

IND = '#ff7f0e'      # promiscuous
DOR = '#000000'      # dormant
UNR = '#c7c7c7'      # unresponsive
SPLIT = 6            # answering this many shocks or more makes a node promiscuous
K = 8

plt.rcParams.update({
  'font.size': 18,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def antimode(B, lo=0.05, hi=0.40):
  c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
  ce = 0.5 * (e[:-1] + e[1:])
  s = np.convolve(c, np.ones(5) / 5, mode='same')
  w = (ce > lo) & (ce < hi)
  return float(ce[w][np.argmin(s[w])])


def load(sens_dir, tag):
  '''Per node breadth n_j, mean sensitivity S, and the cutoff theta.'''
  Sp = np.load(f'{sens_dir}/S-perdrug-rho{tag}.npz')['S']     # (nets, shocks, nodes)
  b = np.load(f'{sens_dir}/B-rho{tag}.npz')
  B, bnets = b['B'], [int(x) for x in b['networks']]
  cut = antimode(B)
  n = (Sp >= cut).sum(axis=1)                                 # (nets, nodes)
  return n, B, bnets, cut, Sp


def classes(n_row, nodes):
  out = []
  for j in nodes:
    k = int(n_row[j])
    out.append('unresponsive' if k == 0 else
               'promiscuous' if k >= SPLIT else 'dormant')
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
  n, B, bnets, cut, _ = load(sens_dir, tag)
  panels = panels_for(ga_csv)
  rows = []
  used = 0
  for net, nodes in panels.items():
    if net not in bnets:
      continue
    bi = bnets.index(net)
    used += 1
    top = list(np.argsort(-B[bi])[:K])
    rnd = list(rng.choice(B.shape[1], K, replace=False))
    for lab, sel in [('evolved', nodes), ('highest sensitivity', top), ('random', rnd)]:
      for c in classes(n[bi], sel):
        rows.append(dict(strategy=lab, cls=c))
  T = pd.DataFrame(rows)
  return (T.groupby(['strategy', 'cls']).size().unstack(fill_value=0) / used
          ).reindex(index=['evolved', 'highest sensitivity', 'random'],
                    columns=['promiscuous', 'dormant', 'unresponsive'], fill_value=0)


def stacked(ax, comp, title=None, legend=False):
  ys = np.arange(len(comp))
  left = np.zeros(len(comp))
  for cls, col in [('promiscuous', IND), ('dormant', DOR), ('unresponsive', UNR)]:
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

  fig = plt.figure(figsize=(15.2, 9.8))
  gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.30)
  ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
  ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

  # a: the breadth distribution at every noise level, on one axis
  greys = ['#c7c7c7', '#7f7f7f', '#222222']
  for tag, eps, col in zip(args.tags, args.eps_labels, greys):
    n, _, _, cut, _ = load(args.sensitivity_dir, tag)
    h = np.bincount(n.ravel(), minlength=11) / n.size
    ax_a.plot(np.arange(11), 100 * h, color=col, lw=2.2, marker='o', markersize=5,
              label=f'$\\varepsilon = {eps}$')
    print(f'eps {eps}: theta {cut:.3f}  unresponsive {100*h[0]:.1f}%  '
          f'dormant {100*h[1:SPLIT].sum():.1f}%  promiscuous {100*h[SPLIT:].sum():.1f}%')
  ax_a.axvline(SPLIT - 0.5, color=IND, lw=1.8, linestyle=(0, (4, 3)))
  ax_a.set_yscale('log')
  ax_a.set_xticks(range(0, 11, 2))
  ax_a.set_xlabel('Shocks answered, $n_j$')
  ax_a.set_ylabel('Percent of nodes')
  ax_a.text(SPLIT - 0.35, 32, 'promiscuous', fontsize=13, color=IND)
  ax_a.legend(frameon=False, fontsize=14, loc='lower left', handlelength=1.2)

  # b: the mean does not determine the breadth
  n_hi, B_hi, _, cut_hi, _ = load(args.sensitivity_dir, args.tags[2])
  ax_b.hist2d(B_hi.ravel(), n_hi.ravel(),
              bins=[np.linspace(0, 0.7, 70), np.arange(-0.5, 11.5, 1)],
              cmap='Greys', norm=matplotlib.colors.LogNorm())
  ax_b.axvline(cut_hi, color=IND, lw=1.8, linestyle=(0, (4, 3)))
  ax_b.axhline(SPLIT - 0.5, color=IND, lw=1.8, linestyle=(0, (4, 3)))
  ax_b.set_xlabel('Mean sensitivity, $S_j$')
  ax_b.set_ylabel('Shocks answered, $n_j$')
  ax_b.set_title(f'$\\varepsilon = {args.eps_labels[2]}$', fontsize=17)
  ax_b.text(cut_hi + 0.012, 8.6, r'$\theta$', fontsize=17, color=IND)
  # the two rules disagree exactly where the mean sits near the cutoff
  agree = ((B_hi.ravel() > cut_hi) == (n_hi.ravel() >= SPLIT)).mean()
  band = (B_hi.ravel() > cut_hi - 0.05) & (B_hi.ravel() < cut_hi + 0.05)
  print(f'mean rule and breadth rule agree on {100*agree:.0f}% of nodes; '
        f'{100*band[((B_hi.ravel() > cut_hi) != (n_hi.ravel() >= SPLIT))].mean():.0f}% '
        f'of the disagreements lie within 0.05 of the cutoff')

  # c: composition by strategy at high noise, the headline panel
  comp_hi = composition(args.sensitivity_dir, args.tags[2], args.ga_csvs[2], rng)
  stacked(ax_c, comp_hi, title=f'$\\varepsilon = {args.eps_labels[2]}$', legend=True)
  ax_c.legend(frameon=False, fontsize=13, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.26), columnspacing=1.2, handlelength=1.1)
  print(comp_hi.round(2).to_string())

  # d: the same across noise, evolved against the heuristic
  width = 0.26
  xs = np.arange(3)
  for j, (lab, col) in enumerate([('evolved', DOR), ('highest sensitivity', IND),
                                  ('random', UNR)]):
    vals = []
    for tag, ga in zip(args.tags, args.ga_csvs):
      c = composition(args.sensitivity_dir, tag, ga, np.random.default_rng(5))
      vals.append(c.loc[lab, 'dormant'])
    ax_d.bar(xs + (j - 1) * width, vals, width * 0.92, color=col,
             edgecolor='#222222', linewidth=1.0, label=lab)
    for x, v in zip(xs + (j - 1) * width, vals):
      ax_d.text(x, v + 0.06, f'{v:.2f}', ha='center', fontsize=12)
  ax_d.set_xticks(xs)
  ax_d.set_xticklabels([f'$\\varepsilon = {e}$' for e in args.eps_labels])
  ax_d.set_ylabel('Dormant members\nper panel')
  ax_d.set_ylim(0, 4.1)
  ax_d.legend(frameon=False, fontsize=12.5, loc='upper left', ncol=1,
              handlelength=1.1, labelspacing=0.3)

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
