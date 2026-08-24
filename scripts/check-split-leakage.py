'''Does splitting train and test by snapshot instead of by trajectory inflate
accuracy?

classifier.get_performance_data draws 50 rows per class for training and tests
on the remaining rows of the same network, without grouping by
initial_condition_idx. Each class holds 10 initial conditions x 10 consecutive
late snapshots, so snapshots from one trajectory land on both sides of the
split. This script measures what that costs.

  A  the pipeline's own split, 50 random rows per class train, the rest test
  B  whole initial conditions held out, 5 per class train and 5 test

Both arms train on 50 rows and test on 50 rows per class, so the only
difference is the grouping. Arm A doubles as a control and must reproduce the
published accuracies. Arm B also sees 5 distinct trajectories instead of 10, so
its drop is an upper bound on the effect of the grouping alone.

Findings when run at m = 8 over 50 networks and 10 splits:

  evolved          eps 0.5  0.960 -> 0.945    eps 1  0.944 -> 0.931
  top sensitivity  eps 0.5  0.714 -> 0.694    eps 1  0.697 -> 0.692
  random           eps 0.5  0.364 -> 0.362    eps 1  0.354 -> 0.351

The ordering between selectors is unchanged. At eps 0 the comparison is empty
by construction: with rho = 1 every replicate is the same initial condition, so
no trajectory can be held out, which the script reports as the number of
distinct initial conditions it found.

Run on the cluster from the repo root:
  python scripts/check-split-leakage.py
'''
import ast
import glob
import multiprocessing
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LEVELS = [('0', 'rho1.0', 'rho1.0'),
          ('0.5', 'rho0.75-b4', 'rho0.75-b4'),
          ('1', 'rho0.5', 'rho0.5')]
NS = '/n/netscratch/nowak/Lab/dbrewster/boolean/drug-rho-sweep'
K, N_SPLITS, SEED = 8, 10, 17
STATES = None


def ga_panels(tag):
    ga = pd.read_csv(f'data/drug-rho-sweep/{tag}/ga-results-clean/combined-full.csv')
    ga = ga[ga.max_num_features == K]
    fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
    out = {}
    for _, r in fin.iterrows():
        nodes = [int(s.split('-')[1]) for s in ast.literal_eval(r['features'])]
        if len(set(nodes)) == K:
            out[int(r['original_network_idx'])] = sorted(set(nodes))
    return out


def vote_accuracy(train, test, feats):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(train[feats], train['Drug'])
    correct = 0
    classes = sorted(test['Drug'].unique())
    for c in classes:
        rows = test[test['Drug'] == c]
        pred = clf.predict(rows[feats])
        vals, cnt = np.unique(pred, return_counts=True)
        correct += int(vals[np.argmax(cnt)] == c)
    return correct / len(classes)


def _one(args):
    net, panels = args
    sub = STATES[STATES.original_network_idx == net]
    rng = np.random.default_rng(SEED + net)
    ics = np.sort(sub.initial_condition_idx.unique())
    out = []
    for name, nodes in panels.items():
        feats = [f'node-{i}' for i in nodes]
        a_scores, b_scores = [], []
        for _ in range(N_SPLITS):
            # A: the pipeline's own split, by snapshot
            tr = sub.groupby('Drug', group_keys=False).apply(
                lambda x: x.sample(n=50, random_state=int(rng.integers(1 << 31))))
            te = sub.drop(tr.index)
            a_scores.append(vote_accuracy(tr, te, feats))
            # B: hold out whole initial conditions
            half = rng.permutation(ics)
            tr_ic, te_ic = set(half[:len(ics) // 2]), set(half[len(ics) // 2:])
            trb = sub[sub.initial_condition_idx.isin(tr_ic)]
            teb = sub[sub.initial_condition_idx.isin(te_ic)]
            b_scores.append(vote_accuracy(trb, teb, feats))
        out.append(dict(net=net, panel=name,
                        by_snapshot=float(np.mean(a_scores)),
                        by_ic=float(np.mean(b_scores))))
    return out


def main():
    global STATES
    for eps, tag, sweep in LEVELS:
        states = sorted(glob.glob(f'{NS}/{sweep}/derived/states-*.csv'))
        if not states:
            print(f'eps {eps}: no states file', flush=True)
            continue
        panels = ga_panels(tag)
        b = np.load(f'data/sensitivity/B-{tag}.npz')
        B, bnets = b['B'], [int(x) for x in b['networks']]
        rng = np.random.default_rng(SEED)
        per_net = {}
        need = set()
        for net, nodes in panels.items():
            top = sorted(np.argsort(-B[bnets.index(net)])[:K].tolist())
            rnd = sorted(rng.choice(B.shape[1], K, replace=False).tolist())
            per_net[net] = {'evolved': nodes, 'top sensitivity': top, 'random': rnd}
            need |= set(nodes) | set(top) | set(rnd)
        cols = ['original_network_idx', 'initial_condition_idx', 'drug_name']
        cols += [f'node-{i}' for i in sorted(need)]
        STATES = pd.read_csv(states[0], usecols=lambda c: c in set(cols))
        STATES = STATES.rename(columns={'drug_name': 'Drug'})

        if eps == '0':   # are the replicate initial conditions distinct at all?
            net0 = sorted(per_net)[0]
            s0 = STATES[(STATES.original_network_idx == net0) & (STATES.Drug == 'control')]
            nodecols = [c for c in STATES.columns if c.startswith('node-')]
            per_ic = s0.groupby('initial_condition_idx')[nodecols].first()
            uniq = per_ic.drop_duplicates().shape[0]
            print(f'eps 0 check: network {net0} control has {per_ic.shape[0]} ICs, '
                  f'{uniq} distinct on the sampled nodes', flush=True)

        with multiprocessing.Pool(16) as pool:
            rows = [r for rs in pool.imap_unordered(
                _one, [(n, per_net[n]) for n in sorted(per_net)]) for r in rs]
        d = pd.DataFrame(rows)
        print(f'--- eps {eps} ({tag}), {d.net.nunique()} networks', flush=True)
        for name, g in d.groupby('panel'):
            drop = g.by_snapshot.mean() - g.by_ic.mean()
            print(f'   {name:16s} by snapshot {g.by_snapshot.mean():.3f} | '
                  f'by held out IC {g.by_ic.mean():.3f} | drop {drop:+.3f}', flush=True)
        d.to_csv(f'siwork/leakage-eps{eps}.csv', index=False)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
