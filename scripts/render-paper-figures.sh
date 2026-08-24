#!/bin/bash
# The verified invocations behind the paper's rendered figures.
#
# Every argument here has been checked against the figure it produces. This file
# exists because the arguments are not recoverable from the figures: one
# re-render of Figure 7 passed the raw clean search outputs instead of the
# rescored sweeps, which cover only m = 8 at eps 0.5 and 1, and the genetic
# algorithm curve silently vanished from 2 of its 3 panels. A second wrong
# argument in the same run dropped the eps 1 random curve. Neither raised an
# error. scripts/check-figure-inputs.py reports which sources are safe, and
# plot-selection-strategies-figure.py now refuses a single size input.
#
# Run on the cluster from the repo root, one target at a time:
#   bash scripts/render-paper-figures.sh fig5
#   bash scripts/render-paper-figures.sh fig7
#   bash scripts/render-paper-figures.sh si
#   bash scripts/render-paper-figures.sh connectivity
set -e
cd "$(dirname "$0")/.."
PY=$PWD/david-brewster-boolean-network-env/bin/python
NS=/n/netscratch/nowak/Lab/dbrewster/boolean
GA0=data/drug-rho-sweep/rho1.0/ga-results-clean/combined-full.csv
GA05=data/drug-rho-sweep/rho0.75-b4/ga-results-clean/combined-full.csv
GA1=data/drug-rho-sweep/rho0.5/ga-results-clean/combined-full.csv

# staging/fig5-sens and staging/fig5-sweep point at the b5 cohort, which is the
# only one consistent end to end at every rho. See the fig5 cohort notes: b1
# class labels are valid only at rho 1.0, 0.99, and 0.5.
case "$1" in

fig5)
  $PY scripts/plot-grand-finding-figure.py \
    --ablation-prefix ablation-k8-deepclean \
    --sensitivity-dir staging/fig5-sens --deep-dir staging/fig5-sens \
    --sweep-dir staging/fig5-sweep \
    --ga-csv-99 staging/fig5-sweep/rho0.99/ga-results/combined-full.csv \
    --ga-csv-50 staging/fig5-sweep/rho0.5/ga-results/combined-full.csv \
    --random-dir-99 staging/fig5-sweep/rho0.99/random-results \
    --random-dir-50 staging/fig5-sweep/rho0.5/random-results \
    --out-dir plots/fig-grand
  ;;

# The GA curves need the rescored sweeps, and the eps 1 random baseline needs
# the original v7 run. Both are the traps described at the top of this file.
fig7)
  $PY scripts/plot-selection-strategies-figure.py \
    --strategies-dirs data/selection-strategies/rho1.0 \
                      data/selection-strategies/rho0.75-b4 \
                      data/selection-strategies/rho0.5 \
    --ga-csvs $NS/rescored/ga-clean-rescored-all-rho1.0.csv \
              $NS/rescored/ga-clean-rescored-all-rho0.75-b4.csv \
              $NS/rescored/ga-clean-rescored-all-rho0.5.csv \
    --random-dirs data/drug-rho-sweep/rho1.0/random-results \
                  data/drug-rho-sweep/rho0.75-b4/random-results \
                  data/drug-fixed-targets-v7/N5000/random-results-v7 \
    --eps-labels 0 0.5 1 \
    --rule-curve-csvs $NS/rescored/rule-prefixes-rho1.0.csv \
                      $NS/rescored/rule-prefixes-rho0.75-b4.csv \
                      $NS/rescored/rule-prefixes-rho0.5.csv \
    --rule-seq-csvs $NS/rescored/rule-sequences-rho1.0.csv \
                    $NS/rescored/rule-sequences-rho0.75-b4.csv \
                    $NS/rescored/rule-sequences-rho0.5.csv \
    --seq-s-files data/sensitivity/S-perdrug-rho1.0.npz \
                  data/sensitivity/S-perdrug-rho0.75-b4.npz \
                  data/sensitivity/S-perdrug-rho0.5.npz \
    --seq-b-files data/sensitivity/B-rho1.0.npz \
                  data/sensitivity/B-rho0.75-b4.npz \
                  data/sensitivity/B-rho0.5.npz \
    --out-dir plots/fig-strategies
  ;;

# staging/si-sens holds activity and redundancy recomputed at eps 0 and 0.5 by
# compute-info-redundancy.py, alongside symlinks to the eps 1 originals.
si)
  $PY scripts/plot-si-figures.py --sensitivity-dir staging/si-sens \
    --sweep-dir staging/fig5-sweep \
    --ga-csv-99 staging/fig5-sweep/rho0.99/ga-results/combined-full.csv \
    --ga-csv-50 staging/fig5-sweep/rho0.5/ga-results/combined-full.csv \
    --convergence-agg-csv $NS/rescored/convergence-agg.csv --out-dir plots/si
  $PY scripts/plot-si-shapley-figure.py --deep-dir staging/fig5-sens \
    --sensitivity-dir data/sensitivity --ga-csv-50 $GA1 --out-dir plots/si
  $PY scripts/plot-si-entropy-figure.py --sensitivity-dir data/sensitivity \
    --ga-csvs $GA0 $GA05 $GA1 --tags 1.0 0.75-b4 0.5 --out-dir plots/si
  $PY scripts/plot-si-coverage-figure.py --sensitivity-dir data/sensitivity \
    --ga-csvs $GA0 $GA05 $GA1 --tags 1.0 0.75-b4 0.5 --out-dir plots/si
  $PY scripts/plot-si-bimodality-figure.py --sensitivity-dir data/sensitivity \
    --rho 0.5 --out-dir plots/si
  $PY scripts/plot-attractor-census.py --data-dir $NS/attractors \
    --t-experiment 1000 --gamma-c 2.09 --out-dir plots/si
  $PY scripts/plot-shock-census-figure.py --census-dir $NS/attractors \
    --halves b --out-dir plots/si
  ;;

# The eps 0.5 column is its own network ensemble, so it needs its own
# connectivity arrays and its own random panel null. See
# scripts/check-cohort-consistency.py.
connectivity)
  $PY scripts/plot-connectivity-figure.py \
    --panel-topology siwork/panel-topology-merged.csv \
    --connectivity rho1.0=data/sensitivity/connectivity-arrays.npz \
                   rho0.5=data/sensitivity/connectivity-arrays.npz \
                   rho0.75=siwork/connectivity-arrays-075b4.npz \
    --sensitivity-dir staging/s12-sens \
    --ga rho1.0=$GA0 rho0.75=$GA05 rho0.5=$GA1 \
    --out-dir plots/fig-connectivity
  ;;

*)
  echo "usage: bash scripts/render-paper-figures.sh {fig5|fig7|si|connectivity}" >&2
  exit 1
  ;;
esac
