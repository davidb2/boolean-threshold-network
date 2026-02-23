#!/usr/bin/env python3
'''
Resubmit only the GA jobs for the N-scaling experiment.
Simulation and extraction already completed successfully.

Usage (from repo root):
  python scripts/resubmit-ga-n-scaling.py
'''
import pathlib
import subprocess
import sys


STATES_FILES = {
  100:  'data/drug-n-scaling/N100/derived/states-1771863139598.csv',
  1000: 'data/drug-n-scaling/N1000/derived/states-1771863139601.csv',
}

NUM_NETWORKS = 50

SLURM_TIME = {100: '3-00:00:00', 1000: '3-00:00:00'}
SLURM_MEM  = {100: '8G',      1000: '16G'}
SLURM_CPUS = {100: 8,         1000: 8}

VENV = 'david-brewster-boolean-network-env/bin/activate'


def sbatch(*, wrap, job_name, time, mem, cpus, output):
  cmd = [
    'sbatch',
    f'--job-name={job_name}',
    f'--time={time}',
    f'--mem={mem}',
    f'--ntasks=1',
    f'--cpus-per-task={cpus}',
    f'--output={output}',
    f'--wrap={wrap}',
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f'ERROR submitting {job_name}:\n{result.stderr}', file=sys.stderr)
    sys.exit(1)
  job_id = result.stdout.strip().split()[-1]
  print(f'  submitted {job_name}: job {job_id}')
  return job_id


def main():
  for N, states_file in STATES_FILES.items():
    ga_out = pathlib.Path(f'data/drug-n-scaling/N{N}/ga-results')
    ga_out.mkdir(parents=True, exist_ok=True)

    log_dir = pathlib.Path(f'slurm/ga/N{N}')
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped = 0
    for i in range(NUM_NETWORKS):
      output_csv = ga_out / f'{i}.csv'
      if output_csv.exists():
        skipped += 1
        continue

      ga_wrap = (
        f'source {VENV} && '
        f'python scripts/genetic-algorithm-selection.py '
        f'--original-network-idx {i} '
        f'--states-file {states_file} '
        f'--network-size {N} '
        f'--output-dir {ga_out}'
      )
      sbatch(
        wrap     = ga_wrap,
        job_name = f'ga-N{N}-{i}',
        time     = SLURM_TIME[N],
        mem      = SLURM_MEM[N],
        cpus     = SLURM_CPUS[N],
        output   = str(log_dir / f'ga-N{N}-{i}-%j.out'),
      )
      submitted += 1

    print(f'N={N}: submitted {submitted}, skipped {skipped} (already done)')

  print('\nMonitor with: squeue -u $USER')


if __name__ == '__main__':
  main()
