#!/usr/bin/env python3
'''Which scoring protocol produced which published accuracy?

Every selector was scored twice, and the 2 protocols disagree by about a point
at eps 0.5, which matters when the main text quotes the search and a rule in the
same sentence.

  sweep prefixes, 10 splits   the protocol main text Figure 7 plots
  m = 8 rescoring, 30 splits  the protocol recorded in rescored/verdicts.txt

Findings at m = 8:

                      sweep                     m8 verdicts
  search              1.0000 / 0.9484 / 0.9373  1.000 / 0.959 / 0.937
  spring rule         0.9673 / 0.9227 / 0.9007  0.967 / 0.923 / 0.901
  information gain    0.9727 / 0.9273 / 0.9164  0.973 / 0.927 / 0.916

The rule and information gain agree to a thousandth across protocols, so only
the search moves. Quoting 1.00, 0.95, 0.94 for the search is the pairing
consistent with the plotted curves.

It also runs the paired comparison of the spring rule against greedy
information gain, which is not significant at any level: p = 0.85, 0.75, 0.18
by Wilcoxon at eps 0, 0.5, 1.

Run on the cluster from the repo root:
  python scripts/check-scoring-protocols.py
'''
import pathlib

import pandas as pd
from scipy import stats

NS = '/n/netscratch/nowak/Lab/dbrewster/boolean'
LEVELS = [('rho1.0', 0), ('rho0.75-b4', 0.5), ('rho0.5', 1)]


def sweep_curve(path):
  d = pd.read_csv(path)
  d = d[d.max_num_features == 8]
  return d.groupby('original_network_idx')['accuracy'].mean()


def strategy_curve(tag, strategy):
  p = pathlib.Path(f'data/selection-strategies/{tag}/{strategy}-results')
  fs = [f for f in sorted(p.glob('*-full.csv')) if f.name != 'combined-full.csv']
  if not fs:
    return None
  d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
  d = d[d.max_num_features == 8]
  return d.groupby('original_network_idx')['accuracy'].mean()


def main():
  print('=== m = 8 means under the sweep protocol')
  for label, fmt in [('spring rule', f'{NS}/rescored/rule-prefixes-{{tag}}.csv'),
                     ('search', f'{NS}/rescored/ga-clean-rescored-all-{{tag}}.csv')]:
    for tag, eps in LEVELS:
      c = sweep_curve(fmt.format(tag=tag))
      print(f'   {label:16s} eps {eps:<4g} {c.mean():.4f} over {len(c)} networks')
  for tag, eps in LEVELS:
    c = strategy_curve(tag, 'infomax')
    if c is not None:
      print(f'   {"information gain":16s} eps {eps:<4g} {c.mean():.4f} over {len(c)} networks')

  print()
  print('=== recorded m8 verdicts, the 30 split protocol')
  v = pathlib.Path(f'{NS}/rescored/verdicts.txt')
  if v.exists():
    print('   ' + v.read_text().strip().replace('\n', '\n   '))

  print()
  print('=== spring rule against greedy information gain, paired over networks')
  for tag, eps in LEVELS:
    r = sweep_curve(f'{NS}/rescored/rule-prefixes-{tag}.csv')
    i = strategy_curve(tag, 'infomax')
    if i is None:
      continue
    common = r.index.intersection(i.index)
    r, i = r[common], i[common]
    w = stats.wilcoxon(r, i)
    t = stats.ttest_rel(r, i)
    print(f'   eps {eps:<4g} rule {r.mean():.4f} infomax {i.mean():.4f} '
          f'diff {r.mean() - i.mean():+.4f} n={len(common)} '
          f'wilcoxon p={w.pvalue:.3f} paired t p={t.pvalue:.3f} '
          f'rule better in {(r > i).sum()}/{len(common)}')


if __name__ == '__main__':
  main()
