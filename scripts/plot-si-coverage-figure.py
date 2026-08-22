#!/usr/bin/env python3
'''Do dormant reporters tile the shock space? (SI figure)

A dormant node answers a minority of the ten shocks, so a panel of them
could in principle divide the shock space between its members, each one
covering what the others miss. This figure asks whether the evolved
panels actually do that, and the answer depends entirely on what they
are compared against.

A node answers shock q when Delta_{j,q} >= theta, with theta the antimode
of the pooled sensitivity histogram. A_j is the set of shocks a node
answers and n_j = |A_j| its breadth: n_j = 0 unresponsive, 1 <= n_j <= 5
dormant, n_j >= 6 promiscuous. Responsive means promiscuous or dormant.

Two nulls are used, and the choice of null is the whole methodological
point.

  matched shock null  each dormant member is replaced by a random subset
                     of the ten shocks of the same size n_j, drawn with
                     each shock's own popularity among that network's
                     dormant nodes. A uniform draw would be wrong: the
                     shocks are not interchangeable (chi square 1875 to
                     3037 on 9 degrees of freedom against uniform), and
                     assuming they are inflates the expected union by
                     about half a shock, which is larger than the effect
                     being measured. This null credits the panel nothing
                     for breadth and asks only whether the answered sets
                     are arranged more complementarily than chance.
  dormant-pool null  each dormant member is replaced by a random node of
                     the same network with the same n_j. It holds the
                     network, the panel size and the breadth multiset
                     fixed and removes only which dormant nodes were
                     chosen. This is the right null for asking what
                     selection did, because breadth is itself selected.

  a  one representative network, chosen as the panel whose union and
     overlap are closest to the medians at that noise level: its dormant
     members (rows) against the ten shocks (columns), shaded by the
     deviation Delta_{j,q}, with a marker on every cell that clears
     theta. The bottom row is the union over the panel's dormant members.
  b  coverage. The distribution over networks of the number of distinct
     shocks answered by at least one dormant member, against both nulls,
     at the highest noise level.
  c  overlap. Mean pairwise Jaccard between the answered sets of the
     dormant members, observed against both nulls, at three noise levels.
     Both nulls agree: the panels overlap less than chance.
  d  the complementarity claim tested directly against the promiscuous
     members. A shock pair (q, q') is confusable when every promiscuous
     member of the panel gives the two shocks the same threshold
     codeword. The specialisation index is the fraction of confusable
     pairs the dormant members break minus the fraction of non-confusable
     pairs they break. It is zero at every noise level, while a
     positive control that picks dormant-class nodes greedily to cover
     the confusable pairs reaches +0.14 to +0.18, so the test has ample
     power.

The summary: evolved panels do spread their dormant members over the
shock space, overlapping less and covering more than chance under either
null, but the effect is modest and coverage is far from complete, about
six of ten shocks. They do not target the shocks the promiscuous members
confuse.

Usage:
  python scripts/plot-si-coverage-figure.py \
    --sensitivity-dir data/sensitivity \
    --ga-csvs <eps0> <eps0.5> <eps1> --tags 1.0 0.75-b4 0.5 \
    --out-dir plots/si-coverage
'''
import argparse
import ast
import itertools
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

IND = '#ff7f0e'      # promiscuous
DOR = '#000000'      # dormant
UNR = '#c7c7c7'      # unresponsive
MID = '#7f7f7f'      # the dormant pool null
SPLIT = 6            # answering this many shocks or more makes a node promiscuous
K = 8
NDRUG = 10
DRAWS = 2000
PAIRS = np.array(list(itertools.combinations(range(NDRUG), 2)))     # 45 shock pairs

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
  '''Per shock deviation, answered sets, breadth, cutoff, network ids.'''
  d = np.load(f'{sens_dir}/S-perdrug-rho{tag}.npz')
  S = d['S'].transpose(0, 2, 1)                        # (nets, nodes, shocks)
  nets = [int(x) for x in d['networks']]
  cut = antimode(np.load(f'{sens_dir}/B-rho{tag}.npz')['B'])
  A = S >= cut                                         # (nets, nodes, shocks)
  return S, A, A.sum(axis=2), cut, nets


def panels_for(ga_csv):
  '''Final generation panel of each network, as a sorted node list.

  The features field is a set literal, so its iteration order is not
  stable between processes; sorting keeps the resampling reproducible.
  '''
  ga = pd.read_csv(ga_csv)
  ga = ga[ga.max_num_features == K]
  fin = ga.loc[ga.groupby('original_network_idx')['generation'].idxmax()]
  out = {}
  for _, r in fin.iterrows():
    nodes = sorted(int(s.split('-')[1]) for s in ast.literal_eval(r['features']))
    if len(set(nodes)) == K:
      out[int(r['original_network_idx'])] = nodes
  return out


def union_jaccard(sets):
  '''Distinct shocks covered, and mean pairwise Jaccard of the answered sets.'''
  u = int(np.any(sets, axis=0).sum())
  js = [(a & b).sum() / (a | b).sum() for a, b in itertools.combinations(sets, 2)]
  return u, float(np.mean(js)) if js else np.nan


def coverage(sens_dir, tag, ga_csv, seed=101, draws=DRAWS):
  '''Observed union and overlap per network against the two nulls.

  Restricted to panels with at least two dormant members, since pairwise
  overlap is undefined below that and union is fixed by n_j alone.
  '''
  _, A, n, cut, nets = load(sens_dir, tag)
  rng = np.random.default_rng(seed)
  rows, null_u = [], {'shock': [], 'pool': []}
  ndorm = []
  for net, nodes in panels_for(ga_csv).items():
    i = nets.index(net)
    dorm = [j for j in nodes if 1 <= n[i, j] <= SPLIT - 1]
    ndorm.append(len(dorm))
    if len(dorm) < 2:
      continue
    ks = [int(n[i, j]) for j in dorm]
    u, jc = union_jaccard([A[i, j] for j in dorm])
    obs_sets = [tuple(A[i, j]) for j in dorm]
    n_pairs = len(dorm) * (len(dorm) - 1) // 2
    ident_obs = sum(obs_sets[a] == obs_sets[b]
                    for a in range(len(dorm)) for b in range(a + 1, len(dorm)))
    # the pool excludes the panel itself, so the null draws other nodes
    other = np.setdiff1d(np.arange(A.shape[1]), nodes)
    pool = {k: other[n[i, other] == k] for k in set(ks)}
    # The ten shocks are NOT interchangeable: some are answered by far more
    # dormant nodes than others, and a uniform null is rejected outright
    # (chi square 1875 to 3037 on 9 degrees of freedom). Drawing shocks
    # uniformly would inflate the expected union by about half a shock,
    # which is larger than the effect being tested, so the shock space null
    # draws each shock with this network's own dormant popularity.
    dpool = other[(n[i, other] >= 1) & (n[i, other] <= SPLIT - 1)]
    w = A[i, dpool].sum(axis=0).astype(float) if len(dpool) else np.ones(NDRUG)
    w = w / w.sum() if w.sum() > 0 else np.full(NDRUG, 1 / NDRUG)
    su, sj, pu, pj, pi = [], [], [], [], []
    for _ in range(draws):
      sets = np.zeros((len(ks), NDRUG), bool)
      for r, k in enumerate(ks):
        sets[r, rng.choice(NDRUG, k, replace=False, p=w)] = True
      a, b = union_jaccard(sets)
      su.append(a)
      sj.append(b)
      draw = [A[i, rng.choice(pool[k])] for k in ks]
      a, b = union_jaccard(draw)
      pu.append(a)
      pj.append(b)
      dsets = [tuple(x) for x in draw]
      pi.append(sum(dsets[q] == dsets[r]
                    for q in range(len(dsets)) for r in range(q + 1, len(dsets))))
    rows.append(dict(net=net, D=len(dorm), union=u, jac=jc,
                     n_pairs=n_pairs, ident=ident_obs, ident_pool=np.mean(pi),
                     union_shock=np.mean(su), jac_shock=np.mean(sj),
                     union_pool=np.mean(pu), jac_pool=np.mean(pj)))
    null_u['shock'] += su
    null_u['pool'] += pu
  return pd.DataFrame(rows), null_u, cut, np.array(ndorm)


def wilcox(a, b):
  d = np.asarray(a) - np.asarray(b)
  return stats.wilcoxon(d).pvalue


def specialisation(sens_dir, tag, ga_csv):
  '''Do dormant members break the pairs the promiscuous members confuse?

  A pair is confusable when the promiscuous members give both shocks the
  same threshold codeword. The positive control replaces the dormant
  members with the same number of dormant class nodes of the same
  network, chosen greedily to cover the confusable pairs.
  '''
  _, A, n, _, nets = load(sens_dir, tag)
  brk = lambda M: M[:, PAIRS[:, 0]] != M[:, PAIRS[:, 1]]
  obs, ctl, frac, seen = [], [], [], 0
  for net, nodes in panels_for(ga_csv).items():
    i = nets.index(net)
    dorm = [j for j in nodes if 1 <= n[i, j] <= SPLIT - 1]
    prom = [j for j in nodes if n[i, j] >= SPLIT]
    if not dorm or not prom:
      continue
    seen += 1
    conf = ~brk(A[i, prom]).any(axis=0)
    if conf.all() or not conf.any():
      continue                                  # index undefined for this panel
    bd = brk(A[i, dorm]).any(axis=0)
    obs.append(bd[conf].mean() - bd[~conf].mean())
    frac.append(conf.mean())
    pool = np.where((n[i] >= 1) & (n[i] <= SPLIT - 1))[0]
    bp = brk(A[i, pool])
    cov, pick = np.zeros(len(PAIRS), bool), []
    for _ in range(len(dorm)):
      gain = (bp[:, conf] & ~cov[conf]).sum(axis=1)
      k = int(np.argmax(gain))
      pick.append(k)
      cov |= bp[k]
    bc = bp[pick].any(axis=0)
    ctl.append(bc[conf].mean() - bc[~conf].mean())
  return np.array(obs), np.array(ctl), float(np.mean(frac)), seen


def example_panel(sens_dir, tag, ga_csv, cov):
  '''A representative panel: the median number of dormant members, then
  the closest to the median union and the median overlap within that.'''
  S, A, n, cut, nets = load(sens_dir, tag)
  c = cov[cov.D == int(cov.D.median())]
  if len(c) < 3:
    c = cov
  z = ((c.union - c.union.median()) / c.union.std()).abs() + \
      ((c.jac - c.jac.median()) / c.jac.std()).abs()
  net = int(c.loc[z.idxmin(), 'net'])
  i = nets.index(net)
  nodes = panels_for(ga_csv)[net]
  dorm = [j for j in nodes if 1 <= n[i, j] <= SPLIT - 1]
  dorm.sort(key=lambda j: (int(np.argmax(A[i, j])), j))    # by first shock answered
  return net, dorm, S[i, dorm], A[i, dorm], cut


def stars(p):
  return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sensitivity-dir', required=True)
  p.add_argument('--ga-csvs', nargs=3, required=True)
  p.add_argument('--tags', nargs=3, required=True)      # rho tags for eps 0, 0.5, 1
  p.add_argument('--eps-labels', nargs=3, default=['0', '0.5', '1'])
  p.add_argument('--out-dir', required=True)
  args = p.parse_args()

  covs, nulls, cuts = [], [], []
  for tag, ga, e in zip(args.tags, args.ga_csvs, args.eps_labels):
    c, nu, cut, D = coverage(args.sensitivity_dir, tag, ga)
    covs.append(c)
    nulls.append(nu)
    cuts.append(cut)
    print(f'eps {e}: theta {cut:.3f}  panels {len(D)}  dormant members per panel '
          f'{D.mean():.3f}  none in {(D == 0).sum()}  one in {(D == 1).sum()}  '
          f'two or more in {len(c)}')

  fig = plt.figure(figsize=(15.6, 10.6))
  gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
  ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
  ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

  # a: one network, its dormant members against the ten shocks
  net, dorm, D, Ans, cut = example_panel(args.sensitivity_dir, args.tags[2],
                                         args.ga_csvs[2], covs[2])
  grid = np.vstack([D, D.max(axis=0)])
  mark = np.vstack([Ans, Ans.any(axis=0)])
  im = ax_a.imshow(grid, cmap='Greys', vmin=0, vmax=1,
                   aspect='auto', interpolation='nearest')
  yy, xx = np.where(mark)
  ax_a.plot(xx, yy, 'o', color=IND, markersize=9, markeredgecolor='white',
            markeredgewidth=1.0, linestyle='none')
  ax_a.axhline(len(dorm) - 0.5, color='white', lw=4)
  ax_a.set_xticks(range(NDRUG))
  ax_a.set_xticklabels(range(1, NDRUG + 1), fontsize=13)
  ax_a.set_yticks(range(len(dorm) + 1))
  ax_a.set_yticklabels([f'node {j}' for j in dorm] + ['any dormant'], fontsize=13)
  ax_a.get_yticklabels()[-1].set_fontweight('bold')
  ax_a.set_xlabel('Shock\n' + r'(marked: $\Delta_{j,q} \geq \theta$)', fontsize=15)
  ax_a.set_title(f'network {net},  $\\varepsilon = {args.eps_labels[2]}$', fontsize=17)
  for s in ax_a.spines.values():
    s.set_visible(False)
  cb = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.02)
  cb.set_label(r'$\Delta_{j,q}$', fontsize=15)
  cb.ax.tick_params(labelsize=12)
  cb.ax.axhline(cut, color=IND, lw=2)
  cb.ax.text(1.9, cut, r'$\theta$', color=IND, fontsize=15, va='center')
  print(f'a: network {net}, dormant members {dorm}, union '
        f'{int(Ans.any(axis=0).sum())} of {NDRUG} shocks')

  # b: coverage against both nulls at the highest noise level
  c, nu = covs[2], nulls[2]
  bins = np.arange(0.5, NDRUG + 1.5, 1)
  ax_b.hist(c.union, bins=bins, color=DOR, alpha=0.85, density=True,
            label=f'evolved panels ($n = {len(c)}$)')
  for vals, col, lab in [(nu['shock'], UNR, 'matched shock null'),
                         (nu['pool'], MID, 'dormant pool null')]:
    h, _ = np.histogram(vals, bins=bins, density=True)
    ax_b.step(np.arange(1, NDRUG + 1), h, where='mid', color=col, lw=3, label=lab)
  ax_b.set_xlim(1.5, 10.5)
  ax_b.set_ylim(0, 0.335)
  ax_b.set_xticks(range(2, NDRUG + 1))
  ax_b.set_xlabel('Distinct shocks covered')
  ax_b.set_ylabel('Density')
  ax_b.set_title(f'$\\varepsilon = {args.eps_labels[2]}$', fontsize=17)
  ax_b.legend(frameon=False, fontsize=12.5, loc='upper left', handlelength=1.2,
              labelspacing=0.32, borderpad=0.1)
  ps = wilcox(c.union, c.union_shock), wilcox(c.union, c.union_pool)
  ax_b.text(0.97, 0.98,
            f'mean {c.union.mean():.2f}\n'
            f'{c.union_shock.mean():.2f}  ({stars(ps[0])})\n'
            f'{c.union_pool.mean():.2f}  ({stars(ps[1])})',
            transform=ax_b.transAxes, fontsize=12.5, va='top', ha='right',
            linespacing=1.65)

  # c: overlap against both nulls at every noise level
  width, xs = 0.26, np.arange(3)
  for j, (key, col, lab) in enumerate([('jac', DOR, 'evolved panels'),
                                       ('jac_shock', UNR, 'matched shock null'),
                                       ('jac_pool', MID, 'dormant pool null')]):
    v = [c[key].mean() for c in covs]
    e = [c[key].sem() for c in covs]
    ax_c.bar(xs + (j - 1) * width, v, width * 0.92, yerr=e, color=col,
             edgecolor='#222222', linewidth=1.0, label=lab,
             error_kw=dict(ecolor='#222222', lw=1.2, capsize=3))
  for x, c, e in zip(xs, covs, args.eps_labels):
    ps = wilcox(c.jac, c.jac_shock), wilcox(c.jac, c.jac_pool)
    ax_c.text(x - width / 2, 0.300, stars(ps[0]), ha='center', fontsize=12)
    ax_c.text(x + width / 2, 0.320, stars(ps[1]), ha='center', fontsize=12)
    pu = wilcox(c.union, c.union_shock), wilcox(c.union, c.union_pool)
    print(f'eps {e}: union   obs {c.union.mean():.4f}  '
          f'shock null {c.union_shock.mean():.4f} (p={pu[0]:.3g})  '
          f'pool null {c.union_pool.mean():.4f} (p={pu[1]:.3g})')
    io, ip_, npair = c.ident.sum(), c.ident_pool.sum(), c.n_pairs.sum()
    print(f'eps {e}: identical pairs obs {100 * io / npair:.1f}%  '
          f'pool null {100 * ip_ / npair:.1f}%  ({io:.0f}/{npair} pairs)')
    print(f'eps {e}: jaccard obs {c.jac.mean():.4f}  '
          f'shock null {c.jac_shock.mean():.4f} (p={ps[0]:.3g})  '
          f'pool null {c.jac_pool.mean():.4f} (p={ps[1]:.3g})')
  ax_c.set_xticks(xs)
  ax_c.set_xticklabels([f'$\\varepsilon = {e}$' for e in args.eps_labels])
  ax_c.set_ylabel('Mean pairwise Jaccard\nbetween dormant members')
  ax_c.set_ylim(0, 0.365)
  ax_c.legend(frameon=False, fontsize=12.5, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), handlelength=1.1, columnspacing=1.2)

  # d: the direct test against what the promiscuous members confuse
  width = 0.32
  obs_all, ctl_all = [], []
  for tag, ga in zip(args.tags, args.ga_csvs):
    o, ct, fr, seen = specialisation(args.sensitivity_dir, tag, ga)
    obs_all.append(o)
    ctl_all.append(ct)
    e = args.eps_labels[args.tags.index(tag)]
    to, tc = stats.ttest_1samp(o, 0), stats.ttest_1samp(ct, 0)
    print(f'eps {e}: usable panels {seen}, index defined on {len(o)}; '
          f'confusable pairs {100*fr:.1f}%; specialisation {o.mean():+.4f} '
          f'(SE {o.std(ddof=1)/np.sqrt(len(o)):.4f}, t={to.statistic:.2f}, '
          f'p={to.pvalue:.3g}); positive control {ct.mean():+.4f} '
          f'(t={tc.statistic:.2f}, p={tc.pvalue:.3g})')
  for j, (vals, col, lab) in enumerate([(obs_all, DOR, 'evolved dormant members'),
                                        (ctl_all, IND, 'positive control')]):
    m = [v.mean() for v in vals]
    e = [1.96 * v.std(ddof=1) / np.sqrt(len(v)) for v in vals]
    ax_d.bar(xs + (j - 0.5) * width, m, width * 0.9, yerr=e, color=col,
             edgecolor='#222222', linewidth=1.0, label=lab,
             error_kw=dict(ecolor='#222222', lw=1.2, capsize=3))
  ax_d.axhline(0, color='#222222', lw=1.0)
  for x, v in zip(xs - 0.5 * width, obs_all):
    top = v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    ax_d.text(x, top + 0.008, f'{v.mean():+.3f}', ha='center', fontsize=12)
  ax_d.set_xticks(xs)
  ax_d.set_xticklabels([f'$\\varepsilon = {e}$' for e in args.eps_labels])
  ax_d.set_ylabel('Specialisation index\n(confusable $-$ other pairs)')
  ax_d.set_ylim(-0.05, 0.245)
  ax_d.legend(frameon=False, fontsize=12.5, loc='upper left', handlelength=1.1,
              labelspacing=0.3, borderpad=0.1)

  for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], 'abcd'):
    ax.text(-0.19, 1.06, letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'si-coverage.svg', bbox_inches='tight')
  fig.savefig(out / 'si-coverage.png', bbox_inches='tight', dpi=300)
  print(f'wrote {out}/si-coverage.svg + .png')


if __name__ == '__main__':
  main()
