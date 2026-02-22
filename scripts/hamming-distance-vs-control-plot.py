import dask.dataframe as dd
import itertools
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import logging

from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import *


logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s :: [%(levelname)-8s] :: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

N = 5_000
SEED: int = 2025
STATES_FILE: Final[str] = 'data/drug-v0127d-N5k-gamma1.6-10drugs/derived/states-1769568223557.csv'
NETWORK_FILE: Final[str] = 'data/drug-v0127d-N5k-gamma1.6-10drugs/derived/networks-1769568223557.csv'
GENETIC_ALGORITHM_RESULTS_FILE: Final[str] = 'data/random-forests/retention-vs-accuracy-v0127d-N5k-gamma1.6-10drugs-genetic-combined.csv'


def main():
  reps = 100
  states_df = pd.read_csv(STATES_FILE)#.compute()
  # states_df = pd.read_csv(STATES_FILE, memory_map=True)
  network_df = pd.read_csv(NETWORK_FILE)
  feature_df = pd.read_csv(GENETIC_ALGORITHM_RESULTS_FILE)

  logger.info('loaded dataframes')

  df = states_df#.copy().reset_index(drop=True)
  node_cols = [f'node-{i}' for i in range(N)]

  df['rep'] = (df.index.to_numpy() % reps).astype(np.int64)
  df = df.set_index(['original_network_idx', 'initial_condition_idx', 'rep'], drop=False)

  X = df[node_cols].to_numpy(dtype=np.float64, copy=False)

  is_control = (df['drug_name'] == 'control').to_numpy()
  controls_mat = pd.DataFrame(X[is_control], index=df.index[is_control], columns=node_cols)

  treated_idx = df.index[~is_control]
  treated_mat = pd.DataFrame(X[~is_control], index=treated_idx, columns=node_cols)

  aligned_controls = controls_mat.reindex(treated_mat.index)

  missing = aligned_controls.isna().any(axis=1)
  if missing.any():
    bad = aligned_controls.index[missing].tolist()[:10]
    raise KeyError(f'Missing control for {missing.sum()} treated keys. Example keys: {bad}')

  diff = aligned_controls.to_numpy() - treated_mat.to_numpy()
  a = treated_idx.get_level_values(0).to_numpy(np.int64)
  b = treated_idx.get_level_values(1).to_numpy(np.int64)
  c = treated_idx.get_level_values(2).to_numpy(np.int64)

  A = np.zeros((50, 10, reps, N), dtype=np.float64)
  A[a, b, c, :] = np.abs(diff)
  B = A.mean(axis=2).mean(axis=1)


  logger.info('loaded A and B')


  data = []
  # fine_grained_selected_data = []
  for max_num_features in 2 ** np.arange(0, 8):
    for original_network_idx in range(50):
      nodes_strs = list(eval(feature_df[(feature_df.original_network_idx == original_network_idx) & (feature_df.max_num_features==max_num_features)].features.iloc[0]))
      node_idxs = [int(node_str.split('-')[1]) for node_str in nodes_strs]
      data.append({
        'type': 'selected',
        'original_network_idx': original_network_idx,
        'node_idxs': node_idxs,
        'max_num_features': max_num_features,
        'dist': B[original_network_idx][node_idxs].mean(),
      })
      data.append({
        'type': 'not-selected',
        'original_network_idx': original_network_idx,
        'node_idxs': node_idxs,
        'max_num_features': max_num_features,
        'dist': np.delete(B[original_network_idx], node_idxs).mean(),
      })
      # for node_idx in node_idxs:
      #   fine_grained_selected_data.append({
      #     'original_network_idx': original_network_idx,
      #     'node_idx': node_idx,
      #     'max_num_features': max_num_features,
      #     'dist': B[original_network_idx][node_idx].squeeze(),
      #     'above-cutoff?': B[original_network_idx][node_idx].squeeze() > 0.0815,
      #   })



  selected_df = pd.DataFrame(data)
  logger.info('loaded selected df')
  df.to_pickle('data/selected-df-gamma1.6.pkl')
  logger.info('stored selected df')
  # fine_grained_selected_df = pd.DataFrame(fine_grained_selected_data)
  

  g = sns.barplot(
    data=selected_df,
    x='max_num_features',
    y='dist',
    hue='type',
  )
  g.set(
    xlabel='Number of features selected',
    ylabel='Average Hamming distance from control',
  )

  g.figure.savefig('plots/hamming-distance-vs-control.png', bbox_inches='tight', dpi=300)



if __name__ == '__main__':
  main()