#!/usr/bin/env python3
'''
Submit SLURM jobs for random node selection using fixed-targets v7 data.

Assumes sim+extract are already complete (states-*.csv exists in derived/).
Submits one job per network -> data/drug-fixed-targets-v7/N5000/random-results-v7/

Completion tracked via {i}-full.done marker files (same convention as GA jobs).

Usage (from repo root):
  python scripts/submit-fixed-targets-v7-random.py
'''
import pathlib
import subprocess
import sys

N            = 5000
NUM_NETWORKS = 50
NUM_WORKERS  = 8

VENV = 'david-brewster-boolean-network-env/bin/activate'

BASE       = pathlib.Path(f'data/drug-fixed-targets-v7/N{N}')
DERIVED    = BASE / 'derived'
RANDOM_OUT = BASE / 'random-results-v7'

SLURM_TIME = '1-00:00:00'
SLURM_MEM  = '16G'
SLURM_CPUS = 8


def slurm_log(network_idx):
    d = pathlib.Path(f'slurm/random/ftN{N}-v7')
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f'random-ftN{N}-v7-{network_idx}-%j.out')


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
    existing_states = list(DERIVED.glob('states-*.csv'))
    if not existing_states:
        print(f'ERROR: no states CSV found in {DERIVED}', file=sys.stderr)
        print('Run sim+extract first (e.g. via submit-fixed-targets-v7.py).', file=sys.stderr)
        sys.exit(1)
    states_file = str(existing_states[0])
    print(f'Using states file: {states_file}')

    RANDOM_OUT.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped   = 0
    for i in range(NUM_NETWORKS):
        done_marker = RANDOM_OUT / f'{i}-full.done'
        if done_marker.exists():
            skipped += 1
            continue

        wrap = (
            f'source {VENV} && '
            f'python scripts/random-node-selection.py '
            f'--original-network-idx {i} '
            f'--states-file {states_file} '
            f'--network-size {N} '
            f'--num-workers {NUM_WORKERS} '
            f'--output-dir {RANDOM_OUT}'
        )
        sbatch(
            wrap     = wrap,
            job_name = f'random-ftN{N}-v7-{i}',
            time     = SLURM_TIME,
            mem      = SLURM_MEM,
            cpus     = SLURM_CPUS,
            output   = slurm_log(i),
        )
        submitted += 1

    print(f'\nSubmitted {submitted} random jobs, skipped {skipped} (already done)')
    print(f'Monitor with: squeue -u $USER')
    print(f'Logs: slurm/random/ftN{N}-v7/')


if __name__ == '__main__':
    main()
