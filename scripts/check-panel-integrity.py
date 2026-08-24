#!/usr/bin/env python3
'''Do the reported panels and ablation cohorts match what the Methods claim?

Two checks that the Methods text depends on.

  1. Every evolved panel carries 8 distinct members. The Methods say panels with
     a repeated member are excluded, which invites the question of how many were.
     The answer is none: all 50 panels at all 3 noise levels have 8 distinct
     nodes, and the feature list is always length 8.
  2. How many independent ablation cohorts exist per noise level. There are 2
     only at eps 0.02, 0.5, and 1, and 1 everywhere else, so a Methods sentence
     claiming 2 at every level is wrong.

Run on the cluster from the repo root:
  python scripts/check-panel-integrity.py
'''
import ast, pathlib
import pandas as pd

print('=== 1. panels with fewer than 8 DISTINCT nodes (clean GA finals)')
for tag, path in [('eps 0   rho1.0    ', 'data/drug-rho-sweep/rho1.0/ga-results-clean/combined-full.csv'),
                  ('eps 0.5 rho0.75-b4', 'data/drug-rho-sweep/rho0.75-b4/ga-results-clean/combined-full.csv'),
                  ('eps 1   rho0.5     ', 'data/drug-rho-sweep/rho0.5/ga-results-clean/combined-full.csv')]:
    ga = pd.read_csv(path)
    ga = ga[ga.max_num_features == 8]
    fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
    rows = []
    for _, r in fin.iterrows():
        nodes = [int(s.split('-')[1]) for s in ast.literal_eval(r['features'])]
        rows.append((int(r['original_network_idx']), len(nodes), len(set(nodes))))
    short = [(n, d) for n, t, d in rows if d < 8]
    print(f'{tag}: {len(rows)} panels, feature list always length 8: '
          f'{all(t == 8 for _, t, _ in rows)}, with <8 distinct: {len(short)} {short[:6]}')

print()
print('=== 2. how many ablation cohorts per noise level (deepclean, as Figure 5 loads them)')
RHOS = ['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.925','0.95','0.975','0.99','0.995','1.0']
VALID_B1 = {'1.0','0.99','0.5'}
d = pathlib.Path('staging/fig5-sens')
for rho in RHOS:
    used = []
    for batch, name in [('b1', f'ablation-k8-deepclean-rho{rho}.csv'),
                        ('b2', f'ablation-k8-deepclean-rho{rho}-b2.csv'),
                        ('b3', f'ablation-k8-deepclean-rho{rho}-b3.csv'),
                        ('b4', f'ablation-k8-deepclean-rho{rho}-b4.csv'),
                        ('b5', f'ablation-k8-deepclean-rho{rho}-b5.csv')]:
        p = d / name
        if not p.exists() or p.stat().st_size < 1000:
            continue
        if batch == 'b1' and rho not in VALID_B1:
            continue
        nets = pd.read_csv(p)['original_network_idx'].nunique()
        used.append(f'{batch}({nets} nets)')
    print(f'  eps {2*(1-float(rho)):<5g} rho {rho:<6}: {len(used)} cohort(s)  {", ".join(used)}')
