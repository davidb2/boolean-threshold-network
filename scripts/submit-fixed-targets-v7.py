#!/usr/bin/env python3
'''
Submit SLURM jobs for the fixed-targets v7 experiment:
  N=5000, IC correlation = 0.5 (vs 0.99 in v5).
  All other settings match v5 (same K_target, targets_per_drug, etc.)

Pipeline:
  1. sim   -> data/drug-fixed-targets-v7/N5000/raw/
  2. extract -> data/drug-fixed-targets-v7/N5000/derived/
  3. GA (50 jobs) -> data/drug-fixed-targets-v7/N5000/ga-results-v7/

Completion tracked via {i}-full.done marker files.
If derived/states-*.csv already exists, sim+extract are skipped.

Usage (from repo root):
  python scripts/submit-fixed-targets-v7.py

SLURM output:
  slurm/sim/sim-ftN5000-v7-{jobid}.out
  slurm/extract/extract-ftN5000-v7-{jobid}.out
  slurm/ga/ftN5000-v7/ga-ftN5000-v7-{i}-{jobid}.out
'''
import pathlib
import subprocess
import sys


# ---------------------------------------------------------------------------
# K_target: average connectivity for gamma=1.8, N=5000 (same as v5)
# ---------------------------------------------------------------------------
def zeta(s, N):
  return sum(k ** (-s) for k in range(1, N + 1))

GAMMA_REF = 1.8
N_REF     = 5000
K_TARGET  = zeta(GAMMA_REF - 1, N_REF) / zeta(GAMMA_REF, N_REF)
print(f'K_target (gamma={GAMMA_REF}, N={N_REF}) = {K_TARGET:.6f}')

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
N                      = 5000
NUM_NETWORKS           = 50
NUM_INITIAL_CONDITIONS = 10
NUM_STEPS              = 1000
NUM_FINAL_STATES       = 10
IC_CORRELATION         = 0.5      # <-- changed from 0.99
NUM_DRUGS              = 10
NUM_TARGETS_PER_DRUG   = 50       # 1% of N=5000
DRUG_STRENGTH          = 1.0
TAG                    = 'fixed-targets-v7'

SLURM_TIME = {'sim': '3-00:00:00', 'extract': '3-00:00:00', 'ga': '3-00:00:00'}
SLURM_MEM  = {'sim': '16G', 'extract': '32G', 'ga': '32G'}
SLURM_CPUS = {'sim': 50,   'extract': 8,     'ga': 16}
GA_WORKERS = 16

VENV = 'david-brewster-boolean-network-env/bin/activate'

BASE    = pathlib.Path(f'data/drug-fixed-targets-v7/N{N}')
RAW     = BASE / 'raw'
DERIVED = BASE / 'derived'
GA_OUT  = BASE / 'ga-results-v7'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slurm_log(stage, network_idx=None):
  if stage == 'ga':
    d = pathlib.Path(f'slurm/ga/ftN{N}-v7')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'ga-ftN{N}-v7-{network_idx}-%j.out')
  else:
    d = pathlib.Path(f'slurm/{stage}')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'{stage}-ftN{N}-v7-%j.out')


def sbatch(*, wrap, job_name, time, mem, cpus, output, dependency=None):
  cmd = [
    'sbatch',
    f'--job-name={job_name}',
    f'--time={time}',
    f'--mem={mem}',
    f'--ntasks=1',
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  for d in [RAW, DERIVED, GA_OUT]:
    d.mkdir(parents=True, exist_ok=True)

  print(f'\n--- N={N}, IC_CORRELATION={IC_CORRELATION} ---')

  # Check if extraction already completed
  existing_states = list(DERIVED.glob('states-*.csv'))
  extract_dep = None

  if existing_states:
    states_ref = str(existing_states[0])
    print(f'  found existing states: {states_ref} — skipping sim+extract')
  else:
    states_ref = f'$(ls {DERIVED}/states-*.csv | head -1)'

    # ------------------------------------------------------------------
    # Step 1: simulation
    # ------------------------------------------------------------------
    sim_cmd = (
      f'RAYON_NUM_THREADS={SLURM_CPUS["sim"]} '
      f'./target/release/perform_experiment '
      f'-n {N} '
      f'-k {K_TARGET} '
      f'--out-degree-distribution power-law '
      f'--num-networks {NUM_NETWORKS} '
      f'--num-initial-conditions {NUM_INITIAL_CONDITIONS} '
      f'--num-steps {NUM_STEPS} '
      f'--num-final-states-to-store {NUM_FINAL_STATES} '
      f'--initial-condition-correlation {IC_CORRELATION} '
      f'--network-seed 0 '
      f'--dynamics-seed 0 '
      f'--num-drugs {NUM_DRUGS} '
      f'--num-targets-per-drug {NUM_TARGETS_PER_DRUG} '
      f'--drug-strength {DRUG_STRENGTH} '
      f'--drug-seed 0 '
      f'--tag {TAG} '
      f'--output-directory {RAW}'
    )
    sim_id = sbatch(
      wrap     = sim_cmd,
      job_name = f'sim-ftN{N}-v7',
      time     = SLURM_TIME['sim'],
      mem      = SLURM_MEM['sim'],
      cpus     = SLURM_CPUS['sim'],
      output   = slurm_log('sim'),
    )

    # ------------------------------------------------------------------
    # Step 2: extract states + networks
    # ------------------------------------------------------------------
    extract_cmd = (
      f'RAYON_NUM_THREADS={SLURM_CPUS["extract"]} '
      f'./target/release/extract_states '
      f'--input-directory {RAW} '
      f'--output-directory {DERIVED}'
    )
    extract_dep = sbatch(
      wrap       = extract_cmd,
      job_name   = f'extract-ftN{N}-v7',
      time       = SLURM_TIME['extract'],
      mem        = SLURM_MEM['extract'],
      cpus       = SLURM_CPUS['extract'],
      output     = slurm_log('extract'),
      dependency = sim_id,
    )

  # ------------------------------------------------------------------
  # Step 3: genetic algorithm — one job per network
  # ------------------------------------------------------------------
  submitted = 0
  skipped   = 0
  ga_ids    = []
  for i in range(NUM_NETWORKS):
    done_marker = GA_OUT / f'{i}-full.done'
    if done_marker.exists():
      skipped += 1
      continue

    ga_wrap = (
      f'source {VENV} && '
      f'python scripts/genetic-algorithm-selection.py '
      f'--original-network-idx {i} '
      f'--states-file {states_ref} '
      f'--network-size {N} '
      f'--num-workers {GA_WORKERS} '
      f'--output-dir {GA_OUT}'
    )
    ga_id = sbatch(
      wrap       = ga_wrap,
      job_name   = f'ga-ftN{N}-v7-{i}',
      time       = SLURM_TIME['ga'],
      mem        = SLURM_MEM['ga'],
      cpus       = SLURM_CPUS['ga'],
      output     = slurm_log('ga', network_idx=i),
      dependency = extract_dep,
    )
    ga_ids.append(ga_id)
    submitted += 1

  if ga_ids:
    print(f'  submitted {submitted} GA jobs (ids {ga_ids[0]}..{ga_ids[-1]}), skipped {skipped}')
  else:
    print(f'  all GA jobs already complete, nothing submitted')

  print('\nAll jobs submitted. Monitor with: squeue -u $USER')
  print(f'Logs: slurm/sim/  slurm/extract/  slurm/ga/ftN{N}-v7/')


if __name__ == '__main__':
  main()
