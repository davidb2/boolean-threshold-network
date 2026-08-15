#!/usr/bin/env python3
'''Submit heuristic selection baselines on the existing v5/v7 datasets.

For each rho in {0.99 (v5), 0.5 (v7)} and each strategy, one array job over the
50 networks runs scripts/heuristic-node-selection.py. Also submits the missing
random baseline for v5 if absent. Everything reads existing derived states, so
no simulation is needed and no bulk data is produced.

Usage (from repo root on the cluster):
  python scripts/submit-heuristics.py
  python scripts/submit-heuristics.py --rhos 0.99 --strategies sensitivity mmse
'''
import argparse
import pathlib
import subprocess
import sys

FULL_GRID    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5000]
JACCARD_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MMSE_GRID    = [1, 2, 4, 8, 16, 32, 64]

ENTROPY_GRID = [1, 2, 4, 8, 16]

STRATEGIES = {
  'sensitivity':       {'grid': FULL_GRID,    'trials': 1, 'time': '12:00:00'},
  'in-degree':         {'grid': FULL_GRID,    'trials': 5, 'time': '1-00:00:00'},
  'out-degree':        {'grid': FULL_GRID,    'trials': 5, 'time': '1-00:00:00'},
  'mmse':              {'grid': MMSE_GRID,    'trials': 1, 'time': '12:00:00'},
  'jaccard':           {'grid': JACCARD_GRID, 'trials': 3, 'time': '1-00:00:00'},
  'influence':         {'grid': JACCARD_GRID, 'trials': 3, 'time': '1-00:00:00'},
  'upstream':          {'grid': JACCARD_GRID, 'trials': 3, 'time': '1-00:00:00'},
  'entropy-diversity': {'grid': MMSE_GRID,    'trials': 1, 'time': '12:00:00'},
  'infomax':           {'grid': ENTROPY_GRID, 'trials': 1, 'time': '12:00:00'},
  'anchor-reporter':   {'grid': ENTROPY_GRID, 'trials': 1, 'time': '12:00:00'},
}

SCRATCH = '/n/netscratch/nowak/Lab/dbrewster/boolean/drug-rho-sweep'

DATASETS = {
  '0.99': {
    'states': 'data/drug-fixed-targets-v5/N5000/derived/states-1771990942417.csv',
    'networks': 'data/drug-fixed-targets-v5/N5000/derived/networks-1771990942417.csv',
    'b': 'data/sensitivity/B-rho0.99.npz',
  },
  '0.5': {
    'states': 'data/drug-fixed-targets-v7/N5000/derived/states-1772488362007.csv',
    'networks': 'data/drug-fixed-targets-v7/N5000/derived/networks-1772488362007.csv',
    'b': 'data/sensitivity/B-rho0.5.npz',
  },
  '1.0': {
    'states': f'$(ls {SCRATCH}/rho1.0/derived/states-*.csv | head -1)',
    'networks': f'$(ls {SCRATCH}/rho1.0/derived/networks-*.csv | head -1)',
    'b': 'data/sensitivity/B-rho1.0.npz',
  },
  '0.75-b4': {
    'states': f'$(ls {SCRATCH}/rho0.75-b4/derived/states-*.csv | head -1)',
    'networks': f'$(ls {SCRATCH}/rho0.75-b4/derived/networks-*.csv | head -1)',
    'b': 'data/sensitivity/B-rho0.75-b4.npz',
  },
}

N            = 5000
NUM_NETWORKS = 50
PARTITION    = 'shared'
VENV         = 'david-brewster-boolean-network-env/bin/activate'
OUT_BASE     = pathlib.Path('data/selection-strategies')


def sbatch(*, wrap, job_name, time, mem, cpus, output, array=None):
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
  if array:
    cmd.append(f'--array={array}%20')
  cmd.append(f'--wrap={wrap}')
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f'ERROR submitting {job_name}:\n{result.stderr}', file=sys.stderr)
    sys.exit(1)
  job_id = result.stdout.strip().split()[-1]
  print(f'  submitted {job_name}: job {job_id}')
  return job_id


def submit(rho, strategy, args):
  cfg = STRATEGIES[strategy]
  paths = DATASETS[rho]
  out_dir = OUT_BASE / f'rho{rho}' / f'{strategy}-results'
  out_dir.mkdir(parents=True, exist_ok=True)
  logs = pathlib.Path(f'slurm/heuristics/rho{rho}')
  logs.mkdir(parents=True, exist_ok=True)

  todo = [i for i in range(NUM_NETWORKS) if not (out_dir / f'{i}-full.done').exists()]
  if not todo:
    print(f'  rho={rho} {strategy}: already complete')
    return

  extra = ''
  if strategy in ('in-degree', 'out-degree', 'jaccard', 'influence', 'upstream'):
    extra = f'--networks-file {paths["networks"]} '
  if strategy in ('sensitivity', 'anchor-reporter', 'infomax'):
    extra = f'--b-file {paths["b"]} '

  wrap = (
    f'source {VENV} && '
    f'python scripts/heuristic-node-selection.py '
    f'--strategy {strategy} '
    f'--original-network-idx ${{SLURM_ARRAY_TASK_ID}} '
    f'--states-file {paths["states"]} '
    f'{extra}'
    f'--network-size {N} '
    f'--feature-sizes {" ".join(map(str, cfg["grid"]))} '
    f'--num-trials {cfg["trials"]} '
    f'--num-workers 8 '
    f'--output-dir {out_dir}'
  )
  sbatch(
    wrap=wrap, job_name=f'{strategy}-rho{rho}', time=cfg['time'],
    mem='32G', cpus=8, output=f'{logs}/{strategy}-%A_%a.out',
    array=','.join(map(str, todo)),
  )


def submit_v5_random(args):
  out_dir = pathlib.Path('data/drug-fixed-targets-v5/N5000/random-results-v5')
  out_dir.mkdir(parents=True, exist_ok=True)
  logs = pathlib.Path('slurm/heuristics/rho0.99')
  logs.mkdir(parents=True, exist_ok=True)
  todo = [i for i in range(NUM_NETWORKS) if not (out_dir / f'{i}-full.done').exists()]
  if not todo:
    print('  v5 random baseline: already complete')
    return
  wrap = (
    f'source {VENV} && '
    f'python scripts/random-node-selection.py '
    f'--original-network-idx ${{SLURM_ARRAY_TASK_ID}} '
    f'--states-file {DATASETS["0.99"]["states"]} '
    f'--network-size {N} '
    f'--num-workers 8 '
    f'--output-dir {out_dir}'
  )
  sbatch(
    wrap=wrap, job_name='rnd-v5', time='1-00:00:00',
    mem='32G', cpus=8, output=f'{logs}/random-%A_%a.out',
    array=','.join(map(str, todo)),
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--rhos', type=str, nargs='+', default=list(DATASETS))
  p.add_argument('--strategies', type=str, nargs='+', default=list(STRATEGIES))
  p.add_argument('--skip-v5-random', action='store_true')
  args = p.parse_args()
  for rho in args.rhos:
    print(f'\n--- rho = {rho} ---')
    for strategy in args.strategies:
      submit(rho, strategy, args)
  if not args.skip_v5_random and 0.99 in args.rhos:
    submit_v5_random(args)
  print('\nAll jobs submitted. Monitor with: squeue -u $USER')


if __name__ == '__main__':
  main()
