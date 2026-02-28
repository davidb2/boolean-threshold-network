#!/usr/bin/env python3
'''
Submit SLURM jobs for the fixed-targets experiment (v5):
  N=500, N=5000, N=50000 at the same <K> as gamma=1.8, N=5000,
  with num_targets_per_drug=1% of N (5 / 50 / 500). Feature sizes are powers
  of 4 (1,4,16,64,...,N) (computed per-N by the GA script).

GA results are written to ga-results-v5/ with per-generation accuracy rows.
Completion is tracked via {i}.done marker files.

If sim+extract already completed for a given N (states-*.csv exists),
those steps are skipped and only GA jobs are submitted.

Usage (from repo root):
  python scripts/submit-fixed-targets.py

SLURM output layout:
  slurm/sim/sim-ftN{N}-{jobid}.out
  slurm/extract/extract-ftN{N}-{jobid}.out
  slurm/ga/ftN{N}/ga-ftN{N}-{i}-{jobid}.out
'''
import pathlib
import subprocess
import sys


# ---------------------------------------------------------------------------
# Compute K_target: average connectivity for gamma=1.8, N=5000
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
# NS = [500, 5000, 50000]
NS = [500, 50000]
# feature sizes are computed per-N by the GA script (powers of 4: 1,4,16,64,..., up to N)

NUM_NETWORKS           = 50
NUM_INITIAL_CONDITIONS = 10
NUM_STEPS              = 1000
NUM_FINAL_STATES       = 10
IC_CORRELATION         = 0.99
NUM_DRUGS              = 10
NUM_TARGETS_PER_DRUG   = {500: 5, 5000: 50, 50000: 500}   # 1% of N
DRUG_STRENGTH          = 1.0
TAG                    = 'fixed-targets-v5'

SLURM_TIME = {
  500:   {'sim': '3-00:00:00', 'extract': '3-00:00:00', 'ga': '3-00:00:00'},
  5000:  {'sim': '3-00:00:00', 'extract': '3-00:00:00', 'ga': '3-00:00:00'},
  50000: {'sim': '3-00:00:00', 'extract': '3-00:00:00', 'ga': '3-00:00:00'},
}
SLURM_MEM = {
  500:   {'sim': '8G',  'extract': '8G',  'ga': '16G'},
  5000:  {'sim': '16G', 'extract': '32G', 'ga': '32G'},
  50000: {'sim': '64G', 'extract': '64G', 'ga': '128G'},
}
# GA_WORKERS controls --num-workers passed to genetic-algorithm-selection.py.
# Each worker runs one RF fit simultaneously, so this is the key memory knob.
# Rule of thumb: num_workers × peak_rf_mem_per_worker < SLURM_MEM[N]['ga']
#   N=500:   32 workers × ~0.3G/worker ≈ 10G  → fits in 16G
#   N=5000:  16 workers × ~1.5G/worker ≈ 24G  → fits in 32G
#   N=50000:  4 workers × ~20G/worker ≈ 80G  → fits in 128G
GA_WORKERS = {500: 32, 5000: 16, 50000: 4}
SLURM_CPUS = {
  500:   {'sim': 50, 'extract': 8, 'ga': 32},
  5000:  {'sim': 50, 'extract': 8, 'ga': 16},
  50000: {'sim': 50, 'extract': 8, 'ga': 4},
}

VENV = 'david-brewster-boolean-network-env/bin/activate'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def data_dir(N):
  return pathlib.Path(f'data/drug-fixed-targets-v5/N{N}')

def slurm_log(stage, N, network_idx=None):
  label = f'ftN{N}'
  if stage == 'ga':
    d = pathlib.Path(f'slurm/ga/{label}')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'ga-{label}-{network_idx}-%j.out')
  else:
    d = pathlib.Path(f'slurm/{stage}')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'{stage}-{label}-%j.out')

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
    ga_out  = base / 'ga-results-v5'
    for d in [raw, derived, ga_out]:
      d.mkdir(parents=True, exist_ok=True)

    print(f'\n--- N={N} (targets_per_drug={NUM_TARGETS_PER_DRUG[N]}) ---')

    # check if extraction already completed
    existing_states = list(derived.glob('states-*.csv'))
    extract_dep = None

    if existing_states:
      states_ref = str(existing_states[0])
      print(f'  found existing states: {states_ref} — skipping sim+extract')
    else:
      states_ref = f'$(ls {derived}/states-*.csv | head -1)'

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
        f'--num-targets-per-drug {NUM_TARGETS_PER_DRUG[N]} '
        f'--drug-strength {DRUG_STRENGTH} '
        f'--drug-seed 0 '
        f'--tag {TAG} '
        f'--output-directory {raw}'
      )
      sim_id = sbatch(
        wrap     = sim_cmd,
        job_name = f'sim-ftN{N}',
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
      extract_dep = sbatch(
        wrap       = extract_cmd,
        job_name   = f'extract-ftN{N}',
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
      done_marker = ga_out / f'{i}-full.done'
      if done_marker.exists():
        continue

      ga_wrap = (
        f'source {VENV} && '
        f'python scripts/genetic-algorithm-selection.py '
        f'--original-network-idx {i} '
        f'--states-file {states_ref} '
        f'--network-size {N} '
        f'--num-workers {GA_WORKERS[N]} '
        f'--output-dir {ga_out}'
      )
      ga_id = sbatch(
        wrap       = ga_wrap,
        job_name   = f'ga-ftN{N}-{i}',
        time       = SLURM_TIME[N]['ga'],
        mem        = SLURM_MEM[N]['ga'],
        cpus       = SLURM_CPUS[N]['ga'],
        output     = slurm_log('ga', N, network_idx=i),
        dependency = extract_dep,
      )
      ga_ids.append(ga_id)

    if ga_ids:
      print(f'  submitted {len(ga_ids)} GA jobs (ids {ga_ids[0]}..{ga_ids[-1]})')
    else:
      print(f'  all GA jobs already complete, nothing submitted')

  print('\nAll jobs submitted. Monitor with: squeue -u $USER')
  print('Logs: slurm/sim/  slurm/extract/  slurm/ga/ftN500/  slurm/ga/ftN5000/  slurm/ga/ftN50000/')


if __name__ == '__main__':
  main()
