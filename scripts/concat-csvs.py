'''Concatenate part CSVs into one file and remove the parts.

Usage:
  python scripts/concat-csvs.py out.csv part1.csv part2.csv ...
'''
import os
import sys

import pandas as pd

out, parts = sys.argv[1], sys.argv[2:]
frames = [pd.read_csv(p) for p in parts if os.path.exists(p)]
if not frames:
  sys.exit(f'no parts found for {out}')
df = pd.concat(frames, ignore_index=True)
df.to_csv(out, index=False)
print(f'wrote {out}  shape={df.shape} from {len(frames)} parts')
for p in parts:
  if os.path.exists(p):
    os.remove(p)
