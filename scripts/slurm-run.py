#!/usr/bin/env python3
import itertools
import subprocess
import tempfile

from dataclasses import dataclass

@dataclass
class ScriptOptions:
  graph_name: str
  birth_mutation_rate: float
  independent_mutation_rate: float
  dynamic: str


def create_temp_script(options: ScriptOptions):
  temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh')
  temp_file.write(
    f"""
#!/bin/sh
OMP_NUM_THREADS=100 \
  ./cmake-build-release/apps/cmake-build-debug/run \
  --N=100 \
  --birth-mutation-rate={options.birth_mutation_rate} \
  --independent-mutation-rate={options.independent_mutation_rate} \
  --num-steps=10000000 \
  --graph-name={options.graph_name} \
  --num-simulations=10000 \
  --dynamic={options.dynamic} \
  --tag=v1.0.0

    """.strip()
  )
  temp_file.close()
  return temp_file.name


def submit_job(script_options, sbatch_args):
  script_path = create_temp_script(script_options)
  sbatch_command = ['sbatch'] + sbatch_args + [script_path]
  result = subprocess.run(sbatch_command, capture_output=True, text=True)

  if result.returncode == 0:
    print("Job submitted successfully.")
    print("Output:", result.stdout)
  else:
    print("Failed to submit job.")
    print("Error:", result.stderr)




def main():
  for (graph_name, birth_mutation_rate, independent_mutation_rate, dynamic) in itertools.product(
    ['complete', 'cycle', 'double-star', 'star'],
    [1e-4, 1e-3, 1e-2],
    [1e-4, 1e-3, 1e-2],
    ['birth-death', 'death-birth'],
  ):
    options = ScriptOptions(
      graph_name=graph_name,
      birth_mutation_rate=birth_mutation_rate,
      independent_mutation_rate=independent_mutation_rate,
      dynamic=dynamic,
    )
    submit_job(script_options=options, sbatch_args=[
      '-J=Equilibrium',
      '--mail-type=ALL',
      '--mail-user=dbrewster@g.harvard.edu',
      '--mem-per-cpu=2G',
      '--time=0-72:00:00',
      '--ntasks=100',
      '--output=./slurm/%j.out',
    ])


if __name__ == '__main__':
  main()
