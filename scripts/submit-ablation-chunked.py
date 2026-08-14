#!/usr/bin/env python3
'''Submit the k=8 node ablation as an array of network chunks.

The serial ablation needs about 24h for 50 networks, so it is split
into chunks of networks that run in parallel and are concatenated
afterwards. Chunk part files double as done markers.

Usage (from repo root on the cluster):
  python scripts/submit-ablation-chunked.py --rho 0.99 \
    --states-file data/drug-fixed-targets-v5/N5000/derived/states-1771990942417.csv \
    --ga-file data/drug-fixed-targets-v5/N5000/ga-results-v5/combined-full.csv \
    --b-file data/sensitivity/B-rho0.99.npz \
    --out data/sensitivity/ablation-k8-rho0.99.csv
'''
import argparse
import pathlib
import subprocess
import sys

NUM_NETWORKS = 50
CHUNK        = 5
PARTITION    = 'shared'
VENV         = 'david-brewster-boolean-network-env/bin/activate'


def sbatch(*, wrap, job_name, time, mem, cpus, output, dependency=None, array=None):
  cmd = [
    'sbatch', f'--partition={PARTITION}', f'--job-name={job_name}',
    f'--time={time}', f'--mem={mem}', '--ntasks=1', f'--cpus-per-task={cpus}',
    f'--output={output}',
  ]
  if array:
    cmd.append(f'--array={array}')
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


def submit(args):
  out = pathlib.Path(args.out)
  logs = pathlib.Path(f'slurm/ablation/rho{args.rho}')
  logs.mkdir(parents=True, exist_ok=True)
  n_chunks = (NUM_NETWORKS + CHUNK - 1) // CHUNK
  part = f'{out}.part${{SLURM_ARRAY_TASK_ID}}'
  lo = f'$((SLURM_ARRAY_TASK_ID * {CHUNK}))'
  hi = f'$((SLURM_ARRAY_TASK_ID * {CHUNK} + {CHUNK - 1}))'

  todo = [i for i in range(n_chunks) if not pathlib.Path(f'{out}.part{i}').exists()]
  if not todo and out.exists():
    print(f'  {out} complete, nothing to do')
    return

  wrap = (
    f'[ -f {part} ] && exit 0; '
    f'source {VENV} && '
    f'python scripts/node-ablation-k8.py '
    f'--rho {args.rho} '
    f'--states-file {args.states_file} '
    f'--ga-file {args.ga_file} '
    f'--b-file {args.b_file} '
    f'--networks {lo}-{hi} '
    f'--out {part}'
  )
  arr_id = sbatch(
    wrap=wrap, job_name=f'abl-rho{args.rho}', time='6:00:00',
    mem='16G', cpus=2, output=f'{logs}/chunk-%A_%a.out',
    array=','.join(map(str, todo)),
  )
  parts = ' '.join(f'{out}.part{i}' for i in range(n_chunks))
  concat_id = sbatch(
    wrap=f'source {VENV} && python scripts/concat-csvs.py {out} {parts}',
    job_name=f'ablcat-rho{args.rho}', time='0:30:00',
    mem='4G', cpus=1, output=f'{logs}/concat-%j.out',
    dependency=arr_id,
  )
  return concat_id


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--rho', type=float, required=True)
  p.add_argument('--states-file', type=str, required=True)
  p.add_argument('--ga-file', type=str, required=True)
  p.add_argument('--b-file', type=str, required=True)
  p.add_argument('--out', type=str, required=True)
  return p.parse_args()


if __name__ == '__main__':
  submit(parse_args())
