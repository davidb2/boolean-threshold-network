#!/usr/bin/env python3
'''Does local wiring predict a node's sensitivity?

Backs the SI claims that sensitivity reflects the dynamical regime rather than
the wiring diagram: that its correlation with in degree is only about +0.06,
that the in degree distribution of this ensemble is narrow, and that frozen
nodes carry far lower sensitivity than active ones.

Degrees come from data/sensitivity/connectivity-arrays.npz, which describes the
batch 1 network ensemble. Topology depends on --network-seed and not on rho, so
that one file is valid for every batch 1 cohort, which
check-cohort-consistency.py confirms for rho 1.0 and rho 0.5. Cohorts b4 and b5
are different ensembles and need their own arrays, so they are only included
when such a file is present.

Correlations are reported pooled over all nodes and networks, and as the mean
over per network correlations, since pooling across networks can manufacture or
hide an association that no single network shows.

Findings:

  in degree      pooled r +0.041, +0.059, +0.062, +0.037 at eps 0, 0.02, 1, and
                 0.5, so the SI figure of +0.06 is the top of the range, and the
                 mean per network correlation is lower still at +0.028 to +0.041
  out degree     pooled r between -0.004 and 0.000 at every level, no
                 association at all, which is the sharper statement because out
                 degree is the power law distributed one: sd 113 and range 1 to
                 4971 against in degree sd 3.6 and range 0 to 32
  frozen nodes   the fraction depends on where the cutoff falls. The SI text
                 range of 44% to 49% corresponds to a cutoff near 0.005, while
                 the bimodality figure splits at 0.02 and gives 51% to 58%. At
                 eps 0 the fraction is 62.5% and flat in the cutoff, because
                 with identical replicates a node either never moves over the
                 stored window or moves a lot
  control        sensitivity against control state variance r = +0.87 to +0.93,
                 which is the claim the activity figure reports

Run on the cluster from the repo root:
  python scripts/check-degree-sensitivity.py
'''
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

# (label, B array, connectivity array, control activity array)
CASES = [
  ('eps 0    rho1.0  b1', 'data/sensitivity/B-rho1.0.npz',
   'data/sensitivity/connectivity-arrays.npz', 'staging/si-sens/activity-rho1.0.npz'),
  ('eps 0.02 rho0.99 b1', 'data/sensitivity/B-rho0.99.npz',
   'data/sensitivity/connectivity-arrays.npz', 'data/sensitivity/activity-rho0.99.npz'),
  ('eps 1    rho0.5  b1', 'data/sensitivity/B-rho0.5.npz',
   'data/sensitivity/connectivity-arrays.npz', 'data/sensitivity/activity-rho0.5.npz'),
  ('eps 0.5  rho0.75 b4', 'data/sensitivity/B-rho0.75-b4.npz',
   'siwork/connectivity-arrays-075b4.npz', 'staging/si-sens/activity-rho0.75.npz'),
]
FROZEN_CUT = 0.02      # control state variance below this counts as frozen


def aligned(b_file, conn_file):
  '''Sensitivity and degrees on the networks the 2 files share.'''
  b = np.load(b_file)
  c = np.load(conn_file)
  B, bnets = b['B'], [int(x) for x in b['networks']]
  cnets = [int(x) for x in c['networks']]
  common = [n for n in bnets if n in set(cnets)]
  bi = [bnets.index(n) for n in common]
  ci = [cnets.index(n) for n in common]
  return B[bi], c['in_deg'][ci], c['out_deg'][ci], common


def report(label, B, ind, outd, nets, act_file):
  rows = []
  for name, deg in [('in degree', ind), ('out degree', outd)]:
    pooled_p = stats.pearsonr(deg.ravel(), B.ravel())
    pooled_s = stats.spearmanr(deg.ravel(), B.ravel())
    per_net = [stats.pearsonr(deg[i], B[i]).statistic for i in range(len(nets))]
    rows.append(dict(quantity=name, pooled_r=pooled_p.statistic,
                     pooled_rho=pooled_s.statistic,
                     per_network_r=float(np.mean(per_net)),
                     per_network_sd=float(np.std(per_net))))
  d = pd.DataFrame(rows)
  print(f'--- {label}, {len(nets)} networks, {B.shape[1]} nodes each')
  print(d.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
  print(f'    in degree  mean {ind.mean():.2f} sd {ind.std():.2f} '
        f'range {ind.min()} to {ind.max()}')
  print(f'    out degree mean {outd.mean():.2f} sd {outd.std():.2f} '
        f'range {outd.min()} to {outd.max()}')
  if pathlib.Path(act_file).exists():
    a = np.load(act_file)
    anets = [int(x) for x in a['networks']]
    idx = [anets.index(n) for n in nets if n in set(anets)]
    keep = [k for k, n in enumerate(nets) if n in set(anets)]
    A, Bk = a['activity'][idx], B[keep]
    frozen = A < FROZEN_CUT
    print(f'    frozen in control {100 * frozen.mean():.1f}% of nodes, '
          f'mean sensitivity {Bk[frozen].mean():.3f} frozen against '
          f'{Bk[~frozen].mean():.3f} active')
    print(f'    sensitivity against control variance r = '
          f'{stats.pearsonr(A.ravel(), Bk.ravel()).statistic:+.3f}')
    # the frozen fraction depends on where the cutoff is put, so report the
    # sweep rather than a single number
    sweep = '  '.join(f'{c}: {100 * (A < c).mean():.1f}%'
                      for c in (0.005, 0.01, 0.02, 0.05))
    print(f'    frozen fraction by cutoff   {sweep}')
  else:
    print(f'    (no control activity array at {act_file})')


def main():
  for label, b_file, conn_file, act_file in CASES:
    if not (pathlib.Path(b_file).exists() and pathlib.Path(conn_file).exists()):
      print(f'--- {label}: skipped, missing inputs')
      continue
    B, ind, outd, nets = aligned(b_file, conn_file)
    report(label, B, ind, outd, nets, act_file)
    print()


if __name__ == '__main__':
  main()
