#!/usr/bin/env python3
'''Submit node-ablation-k8 as SLURM jobs (one per rho).

Each job runs scripts/node-ablation-k8.py for one rho: it reads the states CSV
(only the needed node columns), reuses the GA's RandomForest evaluator, and
writes data/sensitivity/ablation-k8-rho{rho}.csv.

Prereqs: the cached sensitivity matrices must already exist
  data/sensitivity/B-rho0.5.npz, data/sensitivity/B-rho0.99.npz
(produced by Section 1 of notebooks/sensitive-nodes-vs-rho.ipynb).

Usage (from repo root):
  python scripts/submit-node-ablation-k8.py

SLURM output: slurm/ablation/ablation-k8-rho{rho}-{jobid}.out
'''
import pathlib
import subprocess
import sys

RHOS = [0.5, 0.99]
N_TRIALS = 10        # RF train/test splits averaged per subset (higher = smoother, slower)
MAX_REMOVE = 3       # remove m = 1..MAX_REMOVE nodes

SLURM_TIME = '12:00:00'
SLURM_MEM = '32G'
SLURM_CPUS = 2       # classifier RF is single-threaded; a couple cpus is plenty

VENV = 'david-brewster-boolean-network-env/bin/activate'


def sbatch(*, wrap, job_name, output):
  cmd = [
    'sbatch',
    f'--job-name={job_name}',
    f'--time={SLURM_TIME}',
    f'--mem={SLURM_MEM}',
    '--ntasks=1',
    f'--cpus-per-task={SLURM_CPUS}',
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
  log_dir = pathlib.Path('slurm/ablation')
  log_dir.mkdir(parents=True, exist_ok=True)

  submitted = skipped = 0
  for rho in RHOS:
    out_csv = pathlib.Path(f'data/sensitivity/ablation-k8-rho{rho}.csv')
    if out_csv.exists():
      print(f'  skip rho={rho}: {out_csv} already exists (delete to rerun)')
      skipped += 1
      continue

    wrap = (
      f'source {VENV} && '
      f'python scripts/node-ablation-k8.py '
      f'--rho {rho} '
      f'--n-trials {N_TRIALS} '
      f'--max-remove {MAX_REMOVE}'
    )
    sbatch(
      wrap=wrap,
      job_name=f'ablation-k8-rho{rho}',
      output=str(log_dir / f'ablation-k8-rho{rho}-%j.out'),
    )
    submitted += 1

  print(f'\nsubmitted {submitted}, skipped {skipped}.')
  print('Monitor with: squeue -u $USER')
  print('Logs: slurm/ablation/')
  print('Output: data/sensitivity/ablation-k8-rho{rho}.csv')


if __name__ == '__main__':
  main()
