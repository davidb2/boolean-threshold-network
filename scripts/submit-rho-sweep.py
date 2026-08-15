#!/usr/bin/env python3
'''Submit the rho sweep for the sensitive/insensitive composition figure.

For each new IC correlation rho (0.5 and 0.99 already exist as v7/v5), runs
the full pipeline with the SAME networks, drugs, and base initial conditions
as v5/v7 (all seeds 0):

  1. sim      -> netscratch raw/            (perform_experiment)
  2. extract  -> netscratch derived/, then deletes raw .pb  (extract_states)
  3. GA k=8   -> data/drug-rho-sweep/rho{r}/ga-results/       (array, 50 nets)
  4. random k=8 -> data/drug-rho-sweep/rho{r}/random-results/ (one job, loops nets)
  5. combine  -> ga-results/combined-full.csv
  6. B array  -> data/sensitivity/B-rho{r}.npz
  7. ablation -> data/sensitivity/ablation-k8-rho{r}.csv
  8. cleanup  -> deletes netscratch derived/

Bulk data stays on netscratch (90 day retention); only small products land in
the home data/ tree. Idempotent: completed stages are skipped on resubmission.

Usage (from repo root on the cluster):
  python scripts/submit-rho-sweep.py               # all rhos
  python scripts/submit-rho-sweep.py --rhos 0.9    # subset
'''
import argparse
import pathlib
import subprocess
import sys


def zeta(s, N):
  return sum(k ** (-s) for k in range(1, N + 1))

GAMMA_REF = 1.8
N_REF     = 5000
K_TARGET  = zeta(GAMMA_REF - 1, N_REF) / zeta(GAMMA_REF, N_REF)

N                      = 5000
NUM_NETWORKS           = 50
NUM_INITIAL_CONDITIONS = 10
NUM_STEPS              = 1000
NUM_FINAL_STATES       = 10
NUM_DRUGS              = 10
NUM_TARGETS_PER_DRUG   = 50
DRUG_STRENGTH          = 1.0
FEATURE_SIZE           = 8

RHOS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.995]

PARTITION = 'shared'
VENV      = 'david-brewster-boolean-network-env/bin/activate'
SCRATCH   = pathlib.Path('/n/netscratch/nowak/Lab/dbrewster/boolean/drug-rho-sweep')
HOME_OUT  = pathlib.Path('data/drug-rho-sweep')
SENS_OUT  = pathlib.Path('data/sensitivity')


def sbatch(*, wrap, job_name, time, mem, cpus, output, dependency=None, array=None):
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


def log_dir(rho, stage):
  d = pathlib.Path(f'slurm/rho-sweep/rho{rho}')
  d.mkdir(parents=True, exist_ok=True)
  return d / stage


def submit_rho(rho, args):
  batch = args.batch
  suffix = '' if batch == 1 else f'-b{batch}'
  seed0 = (batch - 1) * NUM_NETWORKS
  print(f'\n--- rho = {rho} (batch {batch}) ---')
  raw     = SCRATCH / f'rho{rho}{suffix}' / 'raw'
  derived = SCRATCH / f'rho{rho}{suffix}' / 'derived'
  ga_out  = HOME_OUT / f'rho{rho}{suffix}' / 'ga-results'
  rnd_out = HOME_OUT / f'rho{rho}{suffix}' / 'random-results'
  for d in [raw, derived, ga_out, rnd_out, SENS_OUT]:
    d.mkdir(parents=True, exist_ok=True)

  tag = f'-{args.ablation_tag}' if args.ablation_tag else ''
  b_file       = SENS_OUT / f'B-rho{rho}{suffix}.npz'
  ablation_out = SENS_OUT / f'ablation-k8{tag}-rho{rho}{suffix}.csv'
  states_ref   = f'$(ls {derived}/states-*.csv | head -1)'

  existing_states = list(derived.glob('states-*.csv'))
  extract_dep = None

  if existing_states:
    print(f'  found existing states, skipping sim+extract')
  else:
    sim_cmd = (
      f'rm -f {raw}/*.pb && '
      f'RAYON_NUM_THREADS=48 '
      f'./target/release/perform_experiment '
      f'-n {N} '
      f'-k {K_TARGET} '
      f'--out-degree-distribution power-law '
      f'--num-networks {NUM_NETWORKS} '
      f'--num-initial-conditions {NUM_INITIAL_CONDITIONS} '
      f'--num-steps {NUM_STEPS} '
      f'--num-final-states-to-store {NUM_FINAL_STATES} '
      f'--initial-condition-correlation {rho} '
      f'--network-seed {seed0} '
      f'--dynamics-seed {seed0 * NUM_INITIAL_CONDITIONS} '
      f'--num-drugs {NUM_DRUGS} '
      f'--num-targets-per-drug {NUM_TARGETS_PER_DRUG} '
      f'--drug-strength {DRUG_STRENGTH} '
      f'--drug-seed {seed0 * NUM_DRUGS} '
      f'--tag rho-sweep-{rho}{suffix} '
      f'--output-directory {raw}'
    )
    sim_id = sbatch(
      wrap=sim_cmd, job_name=f'sim-rho{rho}', time='1-00:00:00',
      mem='16G', cpus=48, output=f'{log_dir(rho, "sim")}-%j.out',
    )

    extract_cmd = (
      f'RAYON_NUM_THREADS=8 '
      f'./target/release/extract_states '
      f'--input-directory {raw} '
      f'--output-directory {derived} '
      f'&& rm -f {raw}/*.pb'
    )
    extract_dep = sbatch(
      wrap=extract_cmd, job_name=f'extract-rho{rho}', time='12:00:00',
      mem='32G', cpus=8, output=f'{log_dir(rho, "extract")}-%j.out',
      dependency=sim_id,
    )

  todo = [i for i in range(NUM_NETWORKS) if not (ga_out / f'{i}-full.done').exists()]
  ga_dep = None
  if todo:
    ga_wrap = (
      f'[ -f {ga_out}/${{SLURM_ARRAY_TASK_ID}}-full.done ] && exit 0; '
      f'rm -f {ga_out}/${{SLURM_ARRAY_TASK_ID}}-full.csv; '
      f'source {VENV} && '
      f'python scripts/genetic-algorithm-selection.py '
      f'--original-network-idx ${{SLURM_ARRAY_TASK_ID}} '
      f'--states-file {states_ref} '
      f'--network-size {N} '
      f'--feature-sizes {FEATURE_SIZE} '
      f'--num-workers 16 '
      f'--output-dir {ga_out}'
    )
    ga_dep = sbatch(
      wrap=ga_wrap, job_name=f'ga-rho{rho}', time='6:00:00',
      mem='32G', cpus=16, output=f'{log_dir(rho, "ga")}-%A_%a.out',
      dependency=extract_dep, array=','.join(map(str, todo)),
    )
    print(f'  GA array over {len(todo)} networks')
  else:
    print('  GA already complete')

  rnd_todo = [i for i in range(NUM_NETWORKS) if not (rnd_out / f'{i}-full.done').exists()]
  rnd_dep = None
  if rnd_todo:
    loop_body = (
      f'[ -f {rnd_out}/$i-full.done ] && continue; '
      f'rm -f {rnd_out}/$i-full.csv; '
      f'python scripts/random-node-selection.py '
      f'--original-network-idx $i '
      f'--states-file {states_ref} '
      f'--network-size {N} '
      f'--feature-sizes {FEATURE_SIZE} '
      f'--num-workers 8 '
      f'--output-dir {rnd_out}'
    )
    rnd_wrap = (
      f'source {VENV} && '
      f'for i in {" ".join(map(str, rnd_todo))}; do {loop_body}; done'
    )
    rnd_dep = sbatch(
      wrap=rnd_wrap, job_name=f'rnd-rho{rho}', time='12:00:00',
      mem='32G', cpus=8, output=f'{log_dir(rho, "random")}-%j.out',
      dependency=extract_dep,
    )
  else:
    print('  random baseline already complete')

  combine_wrap = (
    f'source {VENV} && '
    f'python scripts/combine-ga-results.py {ga_out}'
  )
  combine_dep = sbatch(
    wrap=combine_wrap, job_name=f'combine-rho{rho}', time='0:30:00',
    mem='4G', cpus=1, output=f'{log_dir(rho, "combine")}-%j.out',
    dependency=ga_dep,
  )

  b_dep = None
  if not b_file.exists():
    b_wrap = (
      f'source {VENV} && '
      f'python scripts/compute-b-array.py '
      f'--states-file {states_ref} '
      f'--network-size {N} '
      f'--out {b_file}'
    )
    b_dep = sbatch(
      wrap=b_wrap, job_name=f'barr-rho{rho}', time='6:00:00',
      mem='64G', cpus=4, output=f'{log_dir(rho, "b-array")}-%j.out',
      dependency=extract_dep,
    )
  else:
    print(f'  {b_file} already exists')

  abl_deps = ':'.join(x for x in [combine_dep, b_dep, extract_dep] if x)
  chunk = 5 if args.ablation_max_remove <= 3 else 2
  n_chunks = (NUM_NETWORKS + chunk - 1) // chunk
  part = f'{ablation_out}.part${{SLURM_ARRAY_TASK_ID}}'
  lo = f'$((SLURM_ARRAY_TASK_ID * {chunk}))'
  hi = f'$((SLURM_ARRAY_TASK_ID * {chunk} + {chunk - 1}))'
  # 55000 = networks x perturbations x ICs x stored steps; refuse to run on a
  # partially written states file (the failure that poisoned the first deep batch)
  expected_rows = NUM_NETWORKS * (NUM_DRUGS + 1) * NUM_INITIAL_CONDITIONS * NUM_FINAL_STATES
  abl_wrap = (
    f'[ -f {part} ] && exit 0; '
    f'[ "$(wc -l < {states_ref})" -gt {expected_rows} ] || '
    f'{{ echo "states file incomplete, refusing"; exit 1; }}; '
    f'source {VENV} && '
    f'python scripts/node-ablation-k8.py '
    f'--rho {rho} '
    f'--states-file {states_ref} '
    f'--ga-file {ga_out}/combined-full.csv '
    f'--b-file {b_file} '
    f'--networks {lo}-{hi} '
    f'--n-trials {args.ablation_trials} '
    f'--max-remove {args.ablation_max_remove} '
    f'--out {part}'
  )
  abl_arr = sbatch(
    wrap=abl_wrap, job_name=f'abl-rho{rho}{suffix}', time='10:00:00',
    mem='16G', cpus=2, output=f'{log_dir(rho, "ablation")}-%A_%a.out',
    dependency=abl_deps or None, array=','.join(map(str, range(n_chunks))),
  )
  parts = ' '.join(f'{ablation_out}.part{i}' for i in range(n_chunks))
  abl_dep = sbatch(
    wrap=f'source {VENV} && python scripts/concat-csvs.py {ablation_out} {parts}',
    job_name=f'ablcat-rho{rho}', time='0:30:00',
    mem='4G', cpus=1, output=f'{log_dir(rho, "ablation-concat")}-%j.out',
    dependency=abl_arr,
  )

  clean_deps = ':'.join(x for x in [abl_dep, rnd_dep] if x)
  sbatch(
    wrap=f'rm -rf {SCRATCH / f"rho{rho}{suffix}"}',
    job_name=f'clean-rho{rho}', time='0:15:00',
    mem='1G', cpus=1, output=f'{log_dir(rho, "cleanup")}-%j.out',
    dependency=clean_deps or None,
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--rhos', type=float, nargs='+', default=RHOS)
  p.add_argument('--batch', type=int, default=1,
                 help='network cohort; batch b uses seeds offset by (b-1)*50 networks')
  p.add_argument('--ablation-trials', type=int, default=10)
  p.add_argument('--ablation-max-remove', type=int, default=3)
  p.add_argument('--ablation-tag', type=str, default='',
                 help='suffix for the ablation output name, e.g. deep')
  args = p.parse_args()
  print(f'K_target (gamma={GAMMA_REF}, N={N_REF}) = {K_TARGET:.6f}')
  for rho in args.rhos:
    submit_rho(rho, args)
  print('\nAll jobs submitted. Monitor with: squeue -u $USER')


if __name__ == '__main__':
  main()
