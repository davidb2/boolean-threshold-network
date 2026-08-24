'''Is the Figure 5c removal asymmetry robust across noise levels and cohorts?

For every valid deep ablation cohort this runs the paired test of main text
Figure 5c: within each panel take the mean accuracy drop over single
promiscuous removals and over single dormant removals, then test their
difference across networks. The b1 whitelist matches
plot-grand-finding-figure.py, since b1 class labels are only valid where the
sweep states were never regenerated.

It also prints the raw per cohort means, which is what a reader would get by
pooling every removal without pairing. That unpaired view is confounded and can
even flip sign, so it is reported only as a contrast.

Findings: all 18 cohort by level combinations give a positive gap, every one
significant at p <= 0.01, and the 2 independent cohorts agree in sign at each of
the 3 levels where both exist (eps 0.02, 0.5, 1). At eps 1 the cohorts give
+0.0161 (p = 0.010) and +0.0283 (p < 0.0001).

Run on the cluster from the repo root:
  python scripts/check-ablation-cohorts.py
'''
import pathlib, re
import numpy as np, pandas as pd
from scipy import stats

D = pathlib.Path('data/sensitivity')
VALID_B1 = {'1.0', '0.99', '0.5'}
rows = []
for p in sorted(D.glob('ablation-k8-deepclean-rho*.csv')):
    m = re.match(r'ablation-k8-deepclean-rho([\d.]+)(?:-(b\d))?\.csv', p.name)
    rho, batch = m.group(1), (m.group(2) or 'b1')
    if batch == 'b1' and rho not in VALID_B1:
        continue
    d = pd.read_csv(p)
    if d.groupby('original_network_idx')['baseline_acc'].first().mean() < 0.7:
        continue
    m1 = d[d.m_removed == 1]
    s = m1[m1.n_sensitive_removed == 1].groupby('original_network_idx')['acc_drop'].mean()
    i = m1[m1.n_sensitive_removed == 0].groupby('original_network_idx')['acc_drop'].mean()
    common = s.index.intersection(i.index)
    diff = (s[common] - i[common]).dropna()
    t = stats.ttest_1samp(diff, 0)
    rows.append(dict(eps=round(2 * (1 - float(rho)), 3), rho=rho, cohort=batch,
                     n=len(diff), prom=s[common].mean(), dorm=i[common].mean(),
                     gap=diff.mean(), sem=diff.sem(), p=t.pvalue,
                     pos=int((diff > 0).sum())))
r = pd.DataFrame(rows).sort_values(['eps', 'cohort'])
pd.set_option('display.width', 200)
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print()
print('=== levels with 2 cohorts: do they agree?')
for eps, g in r.groupby('eps'):
    if len(g) < 2:
        continue
    a, b = g.iloc[0], g.iloc[1]
    print(f'  eps {eps}: {a.cohort} gap {a.gap:+.4f} (p={a.p:.3f})  vs  '
          f'{b.cohort} gap {b.gap:+.4f} (p={b.p:.3f})   '
          f'{"AGREE in sign" if np.sign(a.gap) == np.sign(b.gap) else "DISAGREE in sign"}')

print()
print(f'=== positive gaps: {int((r.gap > 0).sum())} of {len(r)} cohort by level '
      f'combinations, {int((r.p < 0.05).sum())} of them at p < 0.05')

print()
print('=== unpaired per cohort means, the confounded view, for contrast')
for p_ in sorted(D.glob('ablation-k8-deepclean-rho*.csv')):
    m = re.match(r'ablation-k8-deepclean-rho([\d.]+)(?:-(b\d))?\.csv', p_.name)
    rho, batch = m.group(1), (m.group(2) or 'b1')
    if batch == 'b1' and rho not in VALID_B1:
        continue
    d = pd.read_csv(p_)
    m1 = d[d.m_removed == 1]
    prom = m1[m1.n_sensitive_removed == 1]['acc_drop']
    dorm = m1[m1.n_sensitive_removed == 0]['acc_drop']
    print(f'  eps {2 * (1 - float(rho)):<5g} {batch}: promiscuous {prom.mean():.4f} '
          f'(n={len(prom)}) dormant {dorm.mean():.4f} (n={len(dorm)}) '
          f'gap {prom.mean() - dorm.mean():+.4f}')
