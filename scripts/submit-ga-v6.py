#!/usr/bin/env python3
'''
Submit GA-only SLURM jobs for ga-results-v6, reusing derived states from
drug-fixed-targets-v5 for N=500, N=5000, N=50000.

Sim + extraction are already complete; only the GA is submitted here.
Completion is tracked via {i}-full.done marker files in the ga-results-v6 dir.

Usage (from repo root):
  python scripts/submit-ga-v6.py

SLURM output:
  slurm/ga/ftN{N}-v6/ga-ftN{N}-v6-{i}-{jobid}.out
'''
import pathlib
import subprocess
import sys

# ---------------------------------------------------------------------------
# Hardcoded v5 derived states files (reused for v6)
# NOTE: N=50000 has two states files; using the later one (1771993914127).
#       If your v5 analysis used the earlier one (1771991926713), update here.
# ---------------------------------------------------------------------------
STATES_FILES = {
  500:   'data/drug-fixed-targets-v5/N500/derived/states-1771990821578.csv',
  5000:  'data/drug-fixed-targets-v5/N5000/derived/states-1771990942417.csv',
  50000: 'data/drug-fixed-targets-v5/N50000/derived/states-1771993914127.csv',
}

NUM_NETWORKS = 50

SLURM_TIME = {
  500:   '3-00:00:00',
  5000:  '3-00:00:00',
  50000: '3-00:00:00',
}
SLURM_MEM = {
  500:   '16G',
  5000:  '32G',
  50000: '128G',
}
GA_WORKERS = {500: 32, 5000: 16, 50000: 4}
SLURM_CPUS = {500: 32, 5000: 16, 50000: 4}

VENV = 'david-brewster-boolean-network-env/bin/activate'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  for N, states_file in STATES_FILES.items():
    ga_out = pathlib.Path(f'data/drug-fixed-targets-v5/N{N}/ga-results-v6')
    ga_out.mkdir(parents=True, exist_ok=True)

    log_dir = pathlib.Path(f'slurm/ga/ftN{N}-v6')
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n--- N={N} ---')
    submitted = 0
    skipped = 0
    for i in range(NUM_NETWORKS):
      done_marker = ga_out / f'{i}-full.done'
      if done_marker.exists():
        skipped += 1
        continue

      ga_wrap = (
        f'source {VENV} && '
        f'python scripts/genetic-algorithm-selection.py '
        f'--original-network-idx {i} '
        f'--states-file {states_file} '
        f'--network-size {N} '
        f'--num-workers {GA_WORKERS[N]} '
        f'--output-dir {ga_out}'
      )
      sbatch(
        wrap     = ga_wrap,
        job_name = f'ga-ftN{N}-v6-{i}',
        time     = SLURM_TIME[N],
        mem      = SLURM_MEM[N],
        cpus     = SLURM_CPUS[N],
        output   = str(log_dir / f'ga-ftN{N}-v6-{i}-%j.out'),
      )
      submitted += 1

    print(f'  submitted {submitted}, skipped {skipped} (already done)')

  print('\nAll jobs submitted. Monitor with: squeue -u $USER')
  print('Logs: slurm/ga/ftN500-v6/  slurm/ga/ftN5000-v6/  slurm/ga/ftN50000-v6/')


if __name__ == '__main__':
  main()
