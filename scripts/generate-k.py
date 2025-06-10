import numpy as np
import os
import tqdm

N = 5000
c = 0.99

run_release = """
cargo run \
  --package boolean_threshold_network \
  --bin perform_experiment \
  --release \
  -- \
  -n {N} \
  -k {k} \
  --out-degree-distribution homogeneous \
  --num-steps 30 \
  --num-final-states-to-store 1 \
  --network-seed 0 \
  --num-networks 50 \
  --num-initial-conditions 30 \
  --initial-condition-correlation {c} \
  --dynamics-seed 0 \
  --num-drugs 0 \
  --num-targets-per-drug 0 \
  --drug-strength 0 \
  --drug-seed 0 \
  --tag v0.3.0 \
  --output-directory data/no-drug-homogeneous-phase-transition/N{N}-c{c}/raw \
"""


def main():
  os.makedirs(f"data/no-drug-homogeneous-phase-transition/N{N}-c{c}/raw", exist_ok=True)
  for k in tqdm.tqdm(np.linspace(1, 15, 15+14, endpoint=True)):
    if os.system(run_release.format(k=k, N=N, c=c)):
      raise Exception('Something went wrong with the cargo run command') 


if __name__ == "__main__":
  main()