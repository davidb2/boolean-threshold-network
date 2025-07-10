import numpy as np
import os
import tqdm
from multiprocessing import Pool

N = 50_000
c = 0.99

DISTRIBUTION = "homogeneous"

run_release = """
cargo run \
  --package boolean_threshold_network \
  --bin perform_experiment \
  --release \
  -- \
  -n {N} \
  -k {k} \
  --out-degree-distribution {DISTRIBUTION} \
  --num-steps 100 \
  --num-final-states-to-store 1 \
  --network-seed 0 \
  --num-networks 20 \
  --num-initial-conditions 10 \
  --initial-condition-correlation {c} \
  --dynamics-seed 0 \
  --num-drugs 0 \
  --num-targets-per-drug 0 \
  --drug-strength 0 \
  --drug-seed 0 \
  --tag v0.6.0 \
  --output-directory data/no-drug-{DISTRIBUTION}-phase-transition/N{N}-demo/raw \
"""

def main():
  os.makedirs(f"data/no-drug-{DISTRIBUTION}-phase-transition/N{N}-demo/raw", exist_ok=True)
  RANGE = 0
  STEP = 0.5
  with Pool() as pool:
    for _ in tqdm.tqdm(pool.imap(
      os.system, (
        run_release.format(k=k, N=N, c=c, DISTRIBUTION=DISTRIBUTION)
        for k in np.arange(0, RANGE+STEP, STEP)
      )
    ), total=2*RANGE+1): ...


if __name__ == "__main__":
  main()