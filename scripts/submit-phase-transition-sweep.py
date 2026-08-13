#!/usr/bin/env python3
'''Submit the no-drug power-law phase transition sweep (high statistics).

Regenerates the steady-state Hamming distance vs gamma figure data with more
networks per point than the original run. For each N, one sim job loops over
the gamma grid, then analyze_data reduces all .pb files to a single
hamming-distances CSV in the home data/ tree, then raw files are deleted.

  sim (loops gammas) -> netscratch raw/N{N}/
  analyze            -> data/no-drug-power-law-phase-transition/N{N}-v2/derived/
  cleanup            -> deletes netscratch raw/N{N}/

Usage (from repo root on the cluster):
  python scripts/submit-phase-transition-sweep.py
  python scripts/submit-phase-transition-sweep.py --ns 5000
'''
import argparse
import pathlib
import subprocess
import sys

import numpy as np

NS             = [50, 250, 500, 5000]
GAMMAS         = [f'{g:.2f}' for g in np.arange(1.50, 2.81, 0.02)]
NUM_NETWORKS   = 100
NUM_ICS        = 10
NUM_STEPS      = 1000
NUM_FINAL      = 10
IC_CORRELATION = 0.99

PARTITION = 'shared'
SCRATCH   = pathlib.Path('/n/netscratch/nowak/Lab/dbrewster/boolean/no-drug-phase-transition')
HOME_OUT  = pathlib.Path('data/no-drug-power-law-phase-transition')

SIM_TIME  = {50: '12:00:00', 250: '1-00:00:00', 500: '1-00:00:00', 5000: '3-00:00:00'}
SIM_CPUS  = {50: 16, 250: 16, 500: 32, 5000: 48}
ANALYZE_MEM = {50: '8G', 250: '16G', 500: '16G', 5000: '96G'}


def sbatch(*, wrap, job_name, time, mem, cpus, output, dependency=None):
  cmd = [
    'sbatch',
    f'--partition={PARTITION}',
    f'--job-name={job_name}',
    f'--time={time}',
    f'--mem={mem}',
    '--ntasks=1',
    f'--cpus-per-task={cpus}',
    f'--output={output}',
  ]
  if dependency:
    cmd.append(f'--dependency=afterok:{dependency}')
  cmd.append(f'--wrap={wrap}')
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f'ERROR submitting {job_name}:\n{result.stderr}', file=sys.stderr)
    sys.exit(1)
  job_id = result.stdout.strip().split()[-1]
  print(f'  submitted {job_name}: job {job_id}')
  return job_id


def submit_n(n):
  print(f'\n--- N = {n} ({len(GAMMAS)} gammas) ---')
  raw     = SCRATCH / f'N{n}' / 'raw'
  derived = HOME_OUT / f'N{n}-v2' / 'derived'
  logs    = pathlib.Path(f'slurm/phase-transition/N{n}')
  for d in [raw, derived, logs]:
    d.mkdir(parents=True, exist_ok=True)

  if list(derived.glob('hamming-distances-*.csv')):
    print('  derived CSV already exists, skipping')
    return

  cpus = SIM_CPUS[n]
  one_sim = (
    f'./target/release/perform_experiment '
    f'-n {n} '
    f'--gamma $g '
    f'--out-degree-distribution power-law '
    f'--num-networks {NUM_NETWORKS} '
    f'--num-initial-conditions {NUM_ICS} '
    f'--num-steps {NUM_STEPS} '
    f'--num-final-states-to-store {NUM_FINAL} '
    f'--initial-condition-correlation {IC_CORRELATION} '
    f'--network-seed 0 '
    f'--dynamics-seed 0 '
    f'--num-drugs 0 '
    f'--num-targets-per-drug 0 '
    f'--drug-strength 0 '
    f'--drug-seed 0 '
    f'--tag pt-N{n}-g$g '
    f'--output-directory {raw}'
  )
  sim_wrap = (
    f'rm -f {raw}/*.pb && '
    f'export RAYON_NUM_THREADS={cpus} && '
    f'for g in {" ".join(GAMMAS)}; do {one_sim} || exit 1; done'
  )
  sim_id = sbatch(
    wrap=sim_wrap, job_name=f'pt-sim-N{n}', time=SIM_TIME[n],
    mem='16G', cpus=cpus, output=f'{logs}/sim-%j.out',
  )

  analyze_wrap = (
    f'RAYON_NUM_THREADS=8 '
    f'./target/release/analyze_data '
    f'--input-directory {raw} '
    f'--output-directory {derived}'
  )
  analyze_id = sbatch(
    wrap=analyze_wrap, job_name=f'pt-ana-N{n}', time='12:00:00',
    mem=ANALYZE_MEM[n], cpus=8, output=f'{logs}/analyze-%j.out',
    dependency=sim_id,
  )

  sbatch(
    wrap=f'rm -rf {SCRATCH / f"N{n}"}',
    job_name=f'pt-clean-N{n}', time='0:15:00',
    mem='1G', cpus=1, output=f'{logs}/cleanup-%j.out',
    dependency=analyze_id,
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ns', type=int, nargs='+', default=NS)
  args = p.parse_args()
  for n in args.ns:
    submit_n(n)
  print('\nAll jobs submitted. Monitor with: squeue -u $USER')


if __name__ == '__main__':
  main()
