#!/usr/bin/env python3
'''Shock census figure (SI): shocks rebuild the attractor landscape, yet the
control fingerprint still pins it.

  a  largest basin share, control versus the mean over that network's ten
     shocks, one line per network (paired slopegraph)
  b  fraction of initial conditions that recur within the step budget, per
     network and perturbation (tile map; rows sorted by control value)
  c  basin weighted fingerprint distance from each shocked attractor to its
     nearest control attractor, against a shuffled node null and against
     the spacing between control attractors
  d  the same distance restricted to node sets: the frozen core of the
     control landscape, the non frozen nodes, and the matched null for the
     non frozen nodes (the test that pinning is not frozen bookkeeping)
  e  same initial condition memory: how much better the control basin
     label predicts the shocked destination than the largest basin rule
  f  pinned nodes per network: nodes frozen in every shocked attractor
     whose value still differs across attractors or shocks

Every exact control cycle is destroyed by every shock (key survival zero),
so panels c to f measure what remains of the control landscape at the node
level. Statistical unit is the network; halves of the census are pooled by
tagging units. Uses the fingerprint binary (one u8 row of N on fractions
per attractors CSV row, in row order).

Usage:
  python scripts/plot-shock-census-figure.py \
    --census-dir census-data --halves b --out-dir plots/si-shock-census
'''
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N = 5000
ORDERED = '#1f77b4'
MARGINAL = '#d62728'
DRUGS = [f'drug-{k}' for k in range(1, 11)]

plt.rcParams.update({
  'font.size': 18,
  'mathtext.fontset': 'cm',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'svg.fonttype': 'none',
})


def load_half(census_dir, half):
  d = pathlib.Path(census_dir)
  atts = pd.read_csv(d / f'attractors-shock-g1.8-{half}.csv',
                     dtype={'network_idx': np.int32, 'perturbation': str,
                            'attractor_key': np.uint64, 'period': np.int64})
  fp = np.fromfile(d / f'fingerprints-shock-g1.8-{half}.bin',
                   dtype=np.uint8).reshape(len(atts), N)
  ics = pd.read_csv(d / f'ics-shock-g1.8-{half}.csv',
                    dtype={'network_idx': np.int32, 'perturbation': str,
                           'ic_idx': np.int32, 'transient': np.int64,
                           'period': np.int64, 'converged': str,
                           'attractor_key': np.uint64})
  ics['converged'] = ics['converged'].str.lower() == 'true'
  return atts, fp, ics


def chunked_l1(A, B, cols=None):
  '''Mean absolute difference matrix (len(A) x len(B)) over node columns.'''
  if cols is not None:
    A, B = A[:, cols], B[:, cols]
  out = np.empty((len(A), len(B)), dtype=np.float32)
  step = max(1, int(2e8 / (len(B) * A.shape[1] + 1)))
  for i in range(0, len(A), step):
    out[i:i + step] = np.abs(A[i:i + step, None, :] - B[None, :, :]).mean(axis=2)
  return out


def per_network(unit, atts, fp, ics, rng):
  '''All per network statistics for the figure.'''
  g = ics[ics.network_idx == atts.network_idx.iloc[0]] if False else None
  return None


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--census-dir', type=str, required=True)
  p.add_argument('--halves', type=str, nargs='+', default=['b'])
  p.add_argument('--out-dir', type=str, required=True)
  args = p.parse_args()

  rng = np.random.default_rng(7)
  rows, tiles = [], []
  survival_tests, survival_hits = 0, 0

  for half in args.halves:
    atts, FP, ics = load_half(args.census_dir, half)
    conv = ics[ics.converged]
    basin = conv.groupby(['network_idx', 'perturbation', 'attractor_key']).size()
    for n in sorted(atts.network_idx.unique()):
      unit = f'{half}:{n}'
      an = atts[atts.network_idx == n]
      gn = ics[ics.network_idx == n]
      ctrl = gn[gn.perturbation == 'control'].set_index('ic_idx')
      fc_ctrl = ctrl.converged.mean()

      # tile row: convergence per perturbation
      fc = gn.groupby('perturbation').converged.mean()
      tiles.append(dict(unit=unit, fc_ctrl=fc_ctrl,
                        **{p_: fc.get(p_, np.nan) for p_ in ['control'] + DRUGS}))

      # largest basin share (converged denominator), control and drug mean
      def lb(pert):
        s = basin.get((n, pert), None)
        sub = conv[(conv.network_idx == n) & (conv.perturbation == pert)]
        if not len(sub):
          return np.nan
        vc = sub.attractor_key.value_counts()
        return vc.iloc[0] / len(sub)
      lb_ctrl = lb('control')
      lb_drugs = np.nanmean([lb(p_) for p_ in DRUGS])

      # fingerprints
      ci = an.index[an.perturbation == 'control'].values
      C = FP[ci].astype(np.float32) / 255.0
      frozen = np.isin(FP[ci], [0, 255]).all(axis=0)
      same_frozen = frozen & (FP[ci].max(axis=0) == FP[ci].min(axis=0))
      U = ~frozen
      within = np.nan
      if len(ci) > 1:
        dc = chunked_l1(C, C)
        np.fill_diagonal(dc, np.inf)
        within = float(dc.min(axis=1).mean())

      dists, dnull, dS, dU, dUn = [], [], [], [], []
      all_di, per_drug = [], {}
      ctrl_keys = set(an[an.perturbation == 'control'].attractor_key)
      mems = []
      for p_ in DRUGS:
        di = an.index[an.perturbation == p_].values
        if not len(di):
          continue
        per_drug[p_] = di
        all_di.extend(di)
        survival_tests += len(ctrl_keys)
        survival_hits += len(ctrl_keys & set(an.loc[di].attractor_key))
        Dm = FP[di].astype(np.float32) / 255.0
        w = np.array([basin.get((n, p_, atts.attractor_key.loc[i]), 0) for i in di],
                     dtype=float)
        if w.sum() == 0:
          continue
        w /= w.sum()
        dmat = chunked_l1(Dm, C)
        nn = dmat.argmin(axis=1)
        dists.append(float((dmat.min(axis=1) * w).sum()))
        Dn = np.stack([Dm[i, rng.permutation(N)] for i in range(len(di))])
        dmn = chunked_l1(Dn, C)
        dnull.append(float((dmn.min(axis=1) * w).sum()))
        if same_frozen.sum():
          dSm = chunked_l1(Dm, C, np.where(same_frozen)[0])
          dS.append(float((dSm[np.arange(len(di)), nn] * w).sum()))
        if U.sum() >= 10:
          uidx = np.where(U)[0]
          dUm = chunked_l1(Dm, C, uidx)
          dU.append(float((dUm[np.arange(len(di)), nn] * w).sum()))
          Du = Dm.copy()
          for i in range(len(di)):
            Du[i, uidx] = Dm[i, uidx[rng.permutation(len(uidx))]]
          dUn_m = chunked_l1(Du, C, uidx)
          dUn.append(float((dUn_m.min(axis=1) * w).sum()))
        # same IC memory
        d_ = gn[gn.perturbation == p_].set_index('ic_idx')
        both = ctrl.converged & d_.converged
        if both.sum() >= 30 and ctrl[ctrl.converged].attractor_key.nunique() > 1:
          sub = pd.DataFrame({'cb': ctrl.loc[both, 'attractor_key'].values,
                              'db': d_.loc[both, 'attractor_key'].values})
          modal = sub.groupby('cb')['db'].agg(
            lambda s: s.value_counts().iloc[0]).sum() / len(sub)
          null = sub['db'].value_counts().iloc[0] / len(sub)
          mems.append(modal - null)

      # anchors in the shocked landscapes
      n_anchor = n_basin_inf = 0
      if all_di:
        A = FP[np.array(all_di)]
        anchor = np.isin(A, [0, 255]).all(axis=0) & (A.max(axis=0) != A.min(axis=0))
        n_anchor = int(anchor.sum())
        basin_inf = np.zeros(N, bool)
        for p_, di in per_drug.items():
          fpd = FP[di]
          basin_inf |= (fpd.max(axis=0) != fpd.min(axis=0))
        n_basin_inf = int((anchor & basin_inf).sum())

      rows.append(dict(
        unit=unit, fc_ctrl=fc_ctrl, lb_ctrl=lb_ctrl, lb_drugs=lb_drugs,
        within=within, dist=np.nanmean(dists) if dists else np.nan,
        dnull=np.nanmean(dnull) if dnull else np.nan,
        dS=np.nanmean(dS) if dS else np.nan,
        dU=np.nanmean(dU) if dU else np.nan,
        dUn=np.nanmean(dUn) if dUn else np.nan,
        mem=np.nanmean(mems) if mems else np.nan,
        n_anchor=n_anchor, n_basin_inf=n_basin_inf,
      ))
    print(f'half {half}: {len(atts)} attractors, {atts.network_idx.nunique()} networks')

  R = pd.DataFrame(rows)
  T = pd.DataFrame(tiles).sort_values('fc_ctrl', ascending=False)
  print(f'attractor survival: {survival_hits} of {survival_tests} tests')

  fig = plt.figure(figsize=(15.6, 10.2))
  gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.62)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  ax_c = fig.add_subplot(gs[0, 2])
  ax_d = fig.add_subplot(gs[1, 0])
  ax_e = fig.add_subplot(gs[1, 1])
  ax_f = fig.add_subplot(gs[1, 2])

  # a: slopegraph
  for _, r in R.iterrows():
    col = ORDERED if r.fc_ctrl >= 0.5 else MARGINAL
    ax_a.plot([0, 1], [r.lb_ctrl, r.lb_drugs], color=col, lw=1.2, alpha=0.65,
              marker='o', markersize=3.5)
  ax_a.set_xticks([0, 1])
  ax_a.set_xticklabels(['control', 'shocked\n(mean of ten)'])
  ax_a.set_xlim(-0.25, 1.25)
  ax_a.set_ylabel('Largest basin share')
  ax_a.set_ylim(0, 1.02)
  ax_a.plot([], [], color=ORDERED, label='ordered network')
  ax_a.plot([], [], color=MARGINAL, label='marginal network')
  ax_a.legend(frameon=False, fontsize=13, loc='lower left', handlelength=1.2,
              bbox_to_anchor=(0.0, -0.04))

  # b: tile map
  M = T[['control'] + DRUGS].to_numpy(dtype=float)
  im = ax_b.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')
  ax_b.axvline(0.5, color='#333333', lw=1.2)
  ax_b.set_xticks([0, 3, 6, 9])
  ax_b.set_xticklabels(['ctrl', 's3', 's6', 's9'])
  ax_b.set_xlabel('Perturbation')
  ax_b.set_ylabel('Network (sorted)')
  ax_b.set_yticks([])
  cb = plt.colorbar(im, ax=ax_b, fraction=0.040, pad=0.02)
  cb.set_label('Fraction of ICs that recur', fontsize=11)
  cb.ax.tick_params(labelsize=11)

  # c: fingerprint distances
  cols = [('within', 'control to\nnearest\ncontrol'), ('dist', 'shock to\nnearest\ncontrol'),
          ('dnull', 'shuffled\nnull')]
  for j, (k, lab) in enumerate(cols):
    v = R[k].dropna()
    x = np.full(len(v), j, dtype=float) + rng.uniform(-0.10, 0.10, len(v))
    ax_c.scatter(x, v, s=18, color='#444444', alpha=0.6, lw=0)
    ax_c.hlines(v.median(), j - 0.22, j + 0.22, color='#000000', lw=3)
  ax_c.set_xticks(range(3))
  ax_c.set_xticklabels([lab for _, lab in cols], fontsize=13)
  ax_c.set_ylabel('Fingerprint distance')
  ax_c.set_ylim(0, 0.56)

  # d: node set decomposition
  cols = [('dS', 'frozen\ncore'), ('dU', 'non frozen\nnodes'), ('dUn', 'non frozen\nnull')]
  for j, (k, lab) in enumerate(cols):
    v = R[k].dropna()
    x = np.full(len(v), j, dtype=float) + rng.uniform(-0.10, 0.10, len(v))
    ax_d.scatter(x, v, s=18, color='#444444', alpha=0.6, lw=0)
    ax_d.hlines(v.median(), j - 0.22, j + 0.22, color='#000000', lw=3)
  ax_d.set_xticks(range(3))
  ax_d.set_xticklabels([lab for _, lab in cols], fontsize=13)
  ax_d.set_ylabel('Fingerprint distance')
  ax_d.set_ylim(0, 0.56)

  # e: same IC memory
  v = R['mem'].dropna()
  x = rng.uniform(-0.13, 0.13, len(v))
  ax_e.scatter(x, v, s=26, color='#444444', alpha=0.75, lw=0)
  ax_e.hlines(v.median(), -0.24, 0.24, color='#000000', lw=3)
  ax_e.axhline(0, color='#bbbbbb', lw=1.0, linestyle=(0, (3, 3)))
  ax_e.set_xlim(-0.55, 0.55)
  ax_e.set_xticks([])
  ax_e.set_ylabel('Same IC memory beyond\nlargest basin rule')
  ax_e.set_ylim(-0.06, 0.35)

  # f: anchor node counts
  S = R.sort_values('n_anchor', ascending=False).reset_index()
  xs = np.arange(len(S))
  ax_f.bar(xs, S.n_basin_inf, color='#000000', width=0.8,
           label='basin informative')
  ax_f.bar(xs, S.n_anchor - S.n_basin_inf, bottom=S.n_basin_inf,
           color='#9e9e9e', width=0.8, label='shock informative only')
  ax_f.set_yscale('symlog', linthresh=10)
  ax_f.set_yticks([0, 5, 10, 100])
  ax_f.set_yticklabels(['0', '5', '10', '100'])
  ax_f.set_xlabel('Network (sorted)')
  ax_f.set_ylabel('Pinned nodes')
  ax_f.set_xticks([])
  ax_f.legend(frameon=False, fontsize=13, loc='upper right', handlelength=1.1)

  for ax, letter in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], 'abcdef'):
    ax.text(-0.26, 1.06, letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='#222222')

  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / 'si-shock-census.svg', bbox_inches='tight')
  fig.savefig(out / 'si-shock-census.png', bbox_inches='tight', dpi=300)
  R.to_csv(out / 'si-shock-census-data.csv', index=False)
  print(f'wrote {out}/si-shock-census.svg + .png')


if __name__ == '__main__':
  main()
