#!/usr/bin/env python3
'''
Submit SLURM jobs for the N-scaling experiment:
  N=100 and N=1000, both at the same <K> as gamma=1.8, N=5000.

Usage (from repo root):
  cargo build --release
  python scripts/submit-n-scaling.py

Jobs submitted per N:
  1. simulate     -- perform_experiment -> .pb files
  2. extract      -- extract_states     -> states-*.csv, networks-*.csv
  3. ga-N{N}-{i}  -- genetic-algorithm-selection.py, one independent job per network

SLURM output layout:
  slurm/sim/sim-N{N}-{jobid}.out
  slurm/extract/extract-N{N}-{jobid}.out
  slurm/ga/N{N}/ga-N{N}-{i}-{jobid}.out
'''
import pathlib
import subprocess
import sys


# ---------------------------------------------------------------------------
# Compute K_target: average connectivity for gamma=1.8, N=5000
# Formula mirrors Rust utils.rs: K = zeta(gamma-1, N) / zeta(gamma, N)
# ---------------------------------------------------------------------------
def zeta(s, N):
  return sum(k ** (-s) for k in range(1, N + 1))

GAMMA_REF = 1.8
N_REF = 5000
K_TARGET = zeta(GAMMA_REF - 1, N_REF) / zeta(GAMMA_REF, N_REF)

print(f'K_target (gamma={GAMMA_REF}, N={N_REF}) = {K_TARGET:.6f}')

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
NS = [100, 1000]

NUM_NETWORKS           = 50
NUM_INITIAL_CONDITIONS = 10
NUM_STEPS              = 100
NUM_FINAL_STATES       = 10      # store last 10 timesteps per IC
IC_CORRELATION         = 0.99
NUM_DRUGS              = 10
DRUG_STRENGTH          = 1.0
TAG                    = 'n-scaling-v1'

def num_targets(N):
  '''1% of N, minimum 1.'''
  return max(1, int(0.01 * N))

SLURM_TIME = {
  100:  {'sim': '0:30:00', 'extract': '0:30:00', 'ga': '3-00:00:00'},
  1000: {'sim': '4:00:00', 'extract': '1:00:00', 'ga': '3-00:00:00'},
}
SLURM_MEM = {
  100:  {'sim': '4G',  'extract': '8G',  'ga': '8G'},
  1000: {'sim': '16G', 'extract': '32G', 'ga': '16G'},
}
SLURM_CPUS = {
  100:  {'sim': 4, 'extract': 4, 'ga': 8},
  1000: {'sim': 8, 'extract': 4, 'ga': 8},
}

VENV = 'david-brewster-boolean-network-env/bin/activate'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def data_dir(N):
  return pathlib.Path(f'data/drug-n-scaling/N{N}')

def slurm_log(stage, N, network_idx=None):
  '''Return --output path string and create the parent directory.'''
  if stage == 'ga':
    d = pathlib.Path(f'slurm/ga/N{N}')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'ga-N{N}-{network_idx}-%j.out')
  else:
    d = pathlib.Path(f'slurm/{stage}')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'{stage}-N{N}-%j.out')

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
  for N in NS:
    base    = data_dir(N)
    raw     = base / 'raw'
    derived = base / 'derived'
    ga_out  = base / 'ga-results'
    for d in [raw, derived, ga_out]:
      d.mkdir(parents=True, exist_ok=True)

    targets = num_targets(N)
    print(f'\n--- N={N} (num_targets_per_drug={targets}) ---')

    # ------------------------------------------------------------------
    # Step 1: simulation
    # ------------------------------------------------------------------
    sim_cmd = (
      f'RAYON_NUM_THREADS={SLURM_CPUS[N]["sim"]} '
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
      f'--num-targets-per-drug {targets} '
      f'--drug-strength {DRUG_STRENGTH} '
      f'--drug-seed 0 '
      f'--tag {TAG} '
      f'--output-directory {raw}'
    )
    sim_id = sbatch(
      wrap     = sim_cmd,
      job_name = f'sim-N{N}',
      time     = SLURM_TIME[N]['sim'],
      mem      = SLURM_MEM[N]['sim'],
      cpus     = SLURM_CPUS[N]['sim'],
      output   = slurm_log('sim', N),
    )

    # ------------------------------------------------------------------
    # Step 2: extract states + networks
    # ------------------------------------------------------------------
    extract_cmd = (
      f'RAYON_NUM_THREADS={SLURM_CPUS[N]["extract"]} '
      f'./target/release/extract_states '
      f'--input-directory {raw} '
      f'--output-directory {derived}'
    )
    extract_id = sbatch(
      wrap       = extract_cmd,
      job_name   = f'extract-N{N}',
      time       = SLURM_TIME[N]['extract'],
      mem        = SLURM_MEM[N]['extract'],
      cpus       = SLURM_CPUS[N]['extract'],
      output     = slurm_log('extract', N),
      dependency = sim_id,
    )

    # ------------------------------------------------------------------
    # Step 3: genetic algorithm -- one independent job per network
    # ------------------------------------------------------------------
    ga_ids = []
    for i in range(NUM_NETWORKS):
      ga_wrap = (
        f'source {VENV} && '
        f'STATES_FILE=$(ls {derived}/states-*.csv | head -1) && '
        f'python scripts/genetic-algorithm-selection.py '
        f'--original-network-idx {i} '
        f'--states-file $STATES_FILE '
        f'--network-size {N} '
        f'--output-dir {ga_out}'
      )
      ga_id = sbatch(
        wrap       = ga_wrap,
        job_name   = f'ga-N{N}-{i}',
        time       = SLURM_TIME[N]['ga'],
        mem        = SLURM_MEM[N]['ga'],
        cpus       = SLURM_CPUS[N]['ga'],
        output     = slurm_log('ga', N, network_idx=i),
        dependency = extract_id,
      )
      ga_ids.append(ga_id)

    print(f'  submitted {len(ga_ids)} GA jobs for N={N} (ids {ga_ids[0]}..{ga_ids[-1]})')

  print('\nAll jobs submitted. Monitor with: squeue -u $USER')
  print('Logs: slurm/sim/  slurm/extract/  slurm/ga/N100/  slurm/ga/N1000/')


if __name__ == '__main__':
  main()
