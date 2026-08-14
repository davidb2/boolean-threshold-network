#!/usr/bin/env python3
'''The grand-finding figure: evolved reporter panels need both node classes,
and the balance shifts with noise.

  a  classification accuracy of evolved k=8 panels vs random panels,
     as a function of initial condition noise
  b  number of sensitive nodes in evolved panels vs the random expectation
  c  accuracy drop when one sensitive or one insensitive member is removed
  d  the insensitive share of the total removal penalty

x axis is the bit flip probability 1 - rho on a log scale, so noise
increases to the right.

Usage:
  python scripts/plot-grand-finding-figure.py \
    --sensitivity-dir data/sensitivity \
    --sweep-dir data/drug-rho-sweep \
    --ga-csv-99 ... --ga-csv-50 ... \
    --random-dir-99 ... --random-dir-50 ... \
    --out-dir plots/fig-grand
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RHOS = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85',
        '0.9', '0.925', '0.95', '0.975', '0.99', '0.995']
CHANCE = 1 / 11
SENS = '#eb6834'
INSENS = '#2a78d6'
GA = '#0f3560'
GRAY = '#8b93a1'

plt.rcParams.update({
  'font.size': 12,
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


def load_panels(path):
  ga = pd.read_csv(path)
  ga = ga[ga.max_num_features == 8]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  return {int(r['original_network_idx']): [int(s.split('-')[1]) for s in eval(r['features'])]
          for _, r in fin.iterrows() if len(set(eval(r['features']))) == 8}


def load_random_acc(random_dir):
  frames = [pd.read_csv(p) for p in sorted(pathlib.Path(random_dir).glob('*-full.csv'))]
  df = pd.concat(frames, ignore_index=True)
  df = df[df.max_num_features == 8]
  return df.groupby('original_network_idx')['accuracy'].mean()


def collect(args):
  rows = []
  for rho in RHOS:
    b = np.load(f'{args.sensitivity_dir}/B-rho{rho}.npz')
    B, bnets = b['B'], [int(x) for x in b['networks']]
    cut = antimode(B)
    if rho == '0.99':
      ga_csv, rnd_dir = args.ga_csv_99, args.random_dir_99
    elif rho == '0.5':
      ga_csv, rnd_dir = args.ga_csv_50, args.random_dir_50
    else:
      ga_csv = f'{args.sweep_dir}/rho{rho}/ga-results/combined-full.csv'
      rnd_dir = f'{args.sweep_dir}/rho{rho}/random-results'
    panels = load_panels(ga_csv)
    rnd_acc = load_random_acc(rnd_dir)

    abl = pd.read_csv(f'{args.sensitivity_dir}/ablation-k8-rho{rho}.csv')
    m1 = abl[abl.m_removed == 1]
    base = abl.groupby('original_network_idx')['baseline_acc'].first()
    drop_s = m1[m1.n_sensitive_removed == 1].groupby('original_network_idx')['acc_drop'].mean()
    drop_i = m1[m1.n_sensitive_removed == 0].groupby('original_network_idx')['acc_drop'].mean()

    for net, nodes in panels.items():
      bi = bnets.index(net)
      row = dict(
        rho=float(rho), noise=1 - float(rho), network=net,
        n_sens=int((B[bi, nodes] > cut).sum()),
        expect=8 * float((B[bi] > cut).mean()),
      )
      row['ga_acc'] = base.get(net, np.nan)
      row['rnd_acc'] = rnd_acc.get(net, np.nan)
      row['drop_s'] = drop_s.get(net, np.nan)
      row['drop_i'] = drop_i.get(net, np.nan)
      tot = row['drop_s'] + row['drop_i']
      row['share_i'] = row['drop_i'] / tot if tot and tot > 0 else np.nan
      rows.append(row)
  return pd.DataFrame(rows)


def line(ax, df, col, color, label, ls='-'):
  g = df.groupby('noise')[col]
  m, s = g.mean(), g.sem()
  x = m.index.to_numpy()
  ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=color, alpha=0.20, lw=0)
  ax.plot(x, m.to_numpy(), color=color, lw=2.0, label=label, linestyle=ls,
          marker='o', markersize=4.5)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', type=str, required=True)
  p.add_argument('--sweep-dir', type=str, required=True)
  p.add_argument('--ga-csv-99', type=str, required=True)
  p.add_argument('--ga-csv-50', type=str, required=True)
  p.add_argument('--random-dir-99', type=str, required=True)
  p.add_argument('--random-dir-50', type=str, required=True)
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  df = collect(args)

  fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
  fig.subplots_adjust(hspace=0.42, wspace=0.30)
  (ax_a, ax_b), (ax_c, ax_d) = axes

  line(ax_a, df, 'ga_acc', GA, 'evolved panel')
  line(ax_a, df, 'rnd_acc', GRAY, 'random panel')
  ax_a.axhline(CHANCE, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)))
  ax_a.text(0.0055, CHANCE + 0.03, 'chance', fontsize=9.5, color='#999999')
  ax_a.set_ylabel('Classification accuracy ($m = 8$)')
  ax_a.set_ylim(0, 1.02)
  ax_a.legend(frameon=False, fontsize=10, loc='center left')

  line(ax_b, df, 'n_sens', GA, 'evolved panel')
  line(ax_b, df, 'expect', GRAY, 'random expectation')
  ax_b.set_ylabel('Sensitive nodes per panel (of 8)')
  ax_b.set_ylim(0, 8)
  ax_b.legend(frameon=False, fontsize=10, loc='center left')

  line(ax_c, df, 'drop_s', SENS, 'remove one sensitive node')
  line(ax_c, df, 'drop_i', INSENS, 'remove one insensitive node')
  ax_c.set_ylabel('Accuracy drop')
  ax_c.set_ylim(0, 0.19)
  ax_c.legend(frameon=False, fontsize=10, loc='upper right')

  rng = np.random.default_rng(7)
  xs, shares, los, his = [], [], [], []
  for noise, g in df.groupby('noise'):
    ds = g['drop_s'].dropna().to_numpy()
    di = g['drop_i'].dropna().to_numpy()
    shares.append(di.mean() / (ds.mean() + di.mean()))
    boot = []
    for _ in range(2000):
      bs = rng.choice(ds, len(ds)).mean()
      bi = rng.choice(di, len(di)).mean()
      boot.append(bi / (bs + bi))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    xs.append(noise); los.append(lo); his.append(hi)
  ax_d.fill_between(xs, los, his, color=INSENS, alpha=0.20, lw=0)
  ax_d.plot(xs, shares, color=INSENS, lw=2.0, marker='o', markersize=4.5)
  ax_d.axhline(0.5, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)))
  ax_d.text(0.0055, 0.512, 'equal importance', fontsize=9.5, color='#999999')
  ax_d.set_ylabel('Insensitive share of\nremoval penalty')
  ax_d.set_ylim(0.2, 0.56)

  for ax in axes.ravel():
    ax.set_xscale('log')
    ax.set_xlabel('Initial condition noise, $1 - \\rho$')
    ax.set_xlim(0.004, 0.6)
  for ax, letter in zip(axes.ravel(), 'abcd'):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='#222222')

  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'fig-grand-finding.svg', bbox_inches='tight')
  fig.savefig(out_dir / 'fig-grand-finding.png', bbox_inches='tight', dpi=300)
  df.to_csv(out_dir / 'grand-finding-data.csv', index=False)
  print(f'wrote {out_dir}/fig-grand-finding.svg + .png')


if __name__ == '__main__':
  main()
