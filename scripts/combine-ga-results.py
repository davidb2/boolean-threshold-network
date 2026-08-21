#!/usr/bin/env python3
'''Combine all per-network GA result CSVs in a directory into one combined.csv.

Usage:
  python scripts/combine-ga-results.py data/drug-fixed-targets/N5000/ga-results-v3
'''
import argparse
import pathlib
import pandas as pd

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', type=pathlib.Path)
  args = parser.parse_args()

  csvs = sorted(p for p in args.directory.glob('*-full.csv') if p.name != 'combined-full.csv')
  if not csvs:
    print(f'No CSVs found in {args.directory}')
    return

  combined = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)
  # drop stray header rows left by interleaved or requeued writers
  combined = combined[combined['original_network_idx'].astype(str)
                      != 'original_network_idx']
  out = args.directory / 'combined-full.csv'
  combined.to_csv(out, index=False)
  print(f'Combined {len(csvs)} files ({len(combined)} rows) → {out}')

if __name__ == '__main__':
  main()
