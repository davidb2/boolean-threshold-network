import numpy as np
import os
import tqdm

N = 5000
c = 0.99

analyze_data = f"""
cargo run \
  --package boolean_threshold_network \
  --bin analyze_data \
  --release \
  -- \
  --input-directory data/no-drug-homogeneous-phase-transition/N{N}-c{c}/raw \
  --output-directory data/no-drug-homogeneous-phase-transition/N{N}-c{c}/derived \
"""


def main():
  os.makedirs(f"data/no-drug-homogeneous-phase-transition/N{N}-c{c}/derived", exist_ok=True)
  os.system(analyze_data)


if __name__ == "__main__":
  main()