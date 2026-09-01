'''Does the sensitive versus insensitive removal asymmetry survive an initial
condition grouped split?

For every evolved panel each member is removed in turn and the remaining 7 are
scored under both protocols of check-split-leakage.py. The published claim of
main text Figure 5c is that losing one sensitive member costs more than
losing one insensitive member.

The statistic must be paired within a panel. Pooling every removal across
networks lets panel composition and network difficulty confound the two groups
and can reverse the sign, which is what an unpaired pooled average of the same
data does at eps 1.

Findings, paired within panel and averaged across networks:

  eps 0.5  by snapshot +0.0253 (p = 0.0002)   by held out IC +0.0282 (p < 0.0001)
  eps 1    by snapshot +0.0170 (p = 0.007)    by held out IC +0.0164 (p = 0.013)

So the asymmetry is unchanged at eps 1 and slightly larger under the clean
split at eps 0.5. The grouping was not manufacturing it.

Run on the cluster from the repo root:
  python scripts/check-ablation-split.py

The split here is the sensitivity cutoff, is_sens = B[node] > cut. It is not the
breadth rule that defines the promiscuous, dormant and unresponsive classes, so
the premium computed here is not the one plotted in main text Figure 6b.
'''
import ast, glob, multiprocessing
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

NS = '/n/netscratch/nowak/Lab/dbrewster/boolean/drug-rho-sweep'
LEVELS = [('0.5', 'rho0.75-b4'), ('1', 'rho0.5')]
K, N_SPLITS, SEED = 8, 10, 23
STATES = None

def antimode(B, lo=0.05, hi=0.40):
    c, e = np.histogram(B.ravel(), bins=np.linspace(0, 1, 101))
    ce = 0.5 * (e[:-1] + e[1:])
    s = np.convolve(c, np.ones(5) / 5, mode='same')
    w = (ce > lo) & (ce < hi)
    return float(ce[w][np.argmin(s[w])])

def vote_acc(train, test, feats):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(train[feats], train['Drug'])
    classes = sorted(test['Drug'].unique())
    ok = 0
    for c in classes:
        p = clf.predict(test[test['Drug'] == c][feats])
        v, n = np.unique(p, return_counts=True)
        ok += int(v[np.argmax(n)] == c)
    return ok / len(classes)

def _one(task):
    net, nodes, is_sens = task
    sub = STATES[STATES.original_network_idx == net]
    rng = np.random.default_rng(SEED + net)
    ics = np.sort(sub.initial_condition_idx.unique())
    splits = []
    for _ in range(N_SPLITS):
        tr = sub.groupby('Drug', group_keys=False).apply(
            lambda x: x.sample(n=50, random_state=int(rng.integers(1 << 31))))
        perm = rng.permutation(ics)
        trb = sub[sub.initial_condition_idx.isin(set(perm[:len(ics)//2]))]
        teb = sub[sub.initial_condition_idx.isin(set(perm[len(ics)//2:]))]
        splits.append((tr, sub.drop(tr.index), trb, teb))
    full = [f'node-{i}' for i in nodes]
    base_a = np.mean([vote_acc(a, b, full) for a, b, _, _ in splits])
    base_b = np.mean([vote_acc(c, d, full) for _, _, c, d in splits])
    rows = []
    for j, nd in enumerate(nodes):
        feats = [f'node-{i}' for i in nodes if i != nd]
        da = base_a - np.mean([vote_acc(a, b, feats) for a, b, _, _ in splits])
        db = base_b - np.mean([vote_acc(c, d, feats) for _, _, c, d in splits])
        rows.append(dict(net=net, node=nd, sensitive=bool(is_sens[j]),
                         drop_by_snapshot=float(da), drop_by_ic=float(db)))
    return rows

def paired_gap(d):
    '''The statistic Figure 5c plots: per network means, then the difference.'''
    from scipy import stats
    for proto, col in [('by snapshot', 'drop_by_snapshot'),
                       ('by held out IC', 'drop_by_ic')]:
        s = d[d.sensitive].groupby('net')[col].mean()
        i = d[~d.sensitive].groupby('net')[col].mean()
        common = s.index.intersection(i.index)
        diff = (s[common] - i[common]).dropna()
        t = stats.ttest_1samp(diff, 0)
        print(f'   {proto:15s} paired:   sensitive {s[common].mean():.4f} | '
              f'insensitive {i[common].mean():.4f} | gap {diff.mean():+.4f} '
              f'(SEM {diff.sem():.4f}, p = {t.pvalue:.4f}, '
              f'positive in {int((diff > 0).sum())}/{len(diff)})', flush=True)


def main():
    global STATES
    for eps, tag in LEVELS:
        ga = pd.read_csv(f'data/drug-rho-sweep/{tag}/ga-results-clean/combined-full.csv')
        ga = ga[ga.max_num_features == K]
        fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
        b = np.load(f'data/sensitivity/B-{tag}.npz')
        B, bnets = b['B'], [int(x) for x in b['networks']]
        cut = antimode(B)
        tasks, need = [], set()
        for _, r in fin.iterrows():
            net = int(r['original_network_idx'])
            nodes = sorted({int(s.split('-')[1]) for s in ast.literal_eval(r['features'])})
            if len(nodes) != K: continue
            row = B[bnets.index(net)]
            tasks.append((net, nodes, [row[n] > cut for n in nodes]))
            need |= set(nodes)
        cols = {'original_network_idx', 'initial_condition_idx', 'drug_name'}
        cols |= {f'node-{i}' for i in need}
        f = sorted(glob.glob(f'{NS}/{tag}/derived/states-*.csv'))[0]
        STATES = pd.read_csv(f, usecols=lambda c: c in cols).rename(columns={'drug_name': 'Drug'})
        with multiprocessing.Pool(16) as pool:
            rows = [r for rs in pool.imap_unordered(_one, tasks) for r in rs]
        d = pd.DataFrame(rows)
        d.to_csv(f'siwork/ablation-clean-eps{eps}.csv', index=False)
        print(f'--- eps {eps} ({tag}), {d.net.nunique()} networks, cutoff {cut:.3f}', flush=True)
        for proto, col in [('by snapshot', 'drop_by_snapshot'), ('by held out IC', 'drop_by_ic')]:
            p = d[d.sensitive][col].mean(); q = d[~d.sensitive][col].mean()
            print(f'   {proto:15s} unpaired: sensitive {p:.4f} | insensitive {q:.4f} '
                  f'| gap {p - q:+.4f}  (confounded, see the docstring)', flush=True)
        paired_gap(d)
main()
