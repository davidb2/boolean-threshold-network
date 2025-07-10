import numpy as np
import os
import tqdm

N = 50_000
c = 0.99

DISTRIBUTION = "homogeneous"
analyze_data = f"""
  RAYON_NUM_THREADS=1 \
  cargo run \
  --package boolean_threshold_network \
  --bin analyze_data \
  --release \
  -- \
  --input-directory data/no-drug-{DISTRIBUTION}-phase-transition/N{N}-demo/raw \
  --output-directory data/no-drug-{DISTRIBUTION}-phase-transition/N{N}-demo/derived \
"""


def main():
  os.makedirs(f"data/no-drug-{DISTRIBUTION}-phase-transition/N{N}-demo/derived", exist_ok=True)
  os.system(analyze_data)


if __name__ == "__main__":
  main()