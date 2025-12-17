import itertools
import logging
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import random
import seaborn as sns
import tqdm

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from typing import *

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s :: [%(levelname)-8s] :: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

N = 5_000
DEPENDENT_VARIABLES: Final[List[str]] = [f'node-{x}' for x in range(N)]
CONTROL: Final[str] = 'control'
SEED: int = 2025
DATA_FILE: Final[str] = 'data/drug-power-law-phase-transition-max-drug-strength/derived/states-1752795739394.csv'
NETWORKS_FILE: Final[str] = 'data/drug-power-law-phase-transition-max-drug-strength/derived/networks-1752795739394.csv'
TRAIN_SIZES = [50]
TEST_SIZES = TRAIN_SIZES
NUM_PREDICTIONS = 100


def predict_dist(X: pd.DataFrame, *, clf: RandomForestClassifier) -> pd.Series:
  y = clf.predict(X)
  # Plurality vote.
  values, counts = np.unique(y, return_counts=True)
  max_index = np.argmax(counts)
  drug_prediction = values[max_index]
  return drug_prediction


def predict(
  clf: RandomForestClassifier,
  *,
  X_test: pd.DataFrame,
  y_test: np.ndarray,
  drug_actual: str,
  test_size: int,
  n_predictions: int,
):
  predicted, actuals = [], []
  for _ in range(n_predictions):
    X_int = X_test[y_test == drug_actual]
    if len(X_int) < test_size: break
    X = X_int.sample(test_size)
    drug_prediction = predict_dist(X, clf=clf)
    predicted.append(drug_prediction)
    actuals.append(drug_actual)
  return predicted, actuals


def get_performance_data(df: pd.DataFrame, trial: int, train_size: int, test_size: int, *, dep_vars: Optional[List[str]]):
  logger.info(f'START {trial=} {train_size=} {test_size=}')
  performance_data = []
  dependent_variables = dep_vars if dep_vars is not None else DEPENDENT_VARIABLES
  df_train = (
    df.groupby('Drug', group_keys=False)
      .apply(lambda x: x.sample(n=train_size) if len(x) >= train_size else pd.Series(dtype=np.float64))
  )
  df_test = df.drop(df_train.index)
  X_train, y_train = df_train[dependent_variables], df_train['Drug']

  clf = RandomForestClassifier(n_estimators=100)
  clf.fit(X_train, y_train)

  X_test, y_test = df_test[dependent_variables], df_test['Drug']
  drugs: List[str] = sorted(list((df_train['Drug'].unique())))
  performance_data = []
  for drug_actual in drugs:
    predictions, actuals = predict(
      clf,
      X_test=X_test,
      y_test=y_test,
      drug_actual=drug_actual,
      test_size=test_size,
      n_predictions=NUM_PREDICTIONS,
    )
    performance_data_single = [
      {
        'trial': trial,
        'test_size': test_size,
        'train_size': train_size,
        'drug_prediction': drug_prediction,
        'drug_actual': drug_actual,
        'correct': bool(drug_actual == drug_prediction)
      }
      for drug_prediction, drug_actual in zip(predictions, actuals)
    ]
    performance_data.extend(performance_data_single)
    accuracy = sum(
      result['correct']
      for result in performance_data_single
    ) / len(predictions) if predictions else None
    if accuracy is not None:
      logger.info(f'DONE {trial=} {train_size=} {test_size=} {drug_actual=} accuracy={accuracy * 100:.2f}')
  return performance_data, clf


def train_and_test(df: pd.DataFrame, num_trials: int, dep_vars: Optional[List[str]] = None):
  performance_data, clfs = [], []
  for data, clf in (
      get_performance_data(df, trial, train_size, test_size, dep_vars=dep_vars)
        for trial in range(num_trials)
        for train_size in TRAIN_SIZES
        for test_size in TEST_SIZES
      ):
    performance_data.extend(data)
    clfs.append(clf)
  return pd.DataFrame(performance_data), clfs


def plot_confusion(performance_df: pd.DataFrame):
  DRUGS = sorted(list((performance_df['drug_actual'].unique())))
  test_sizes = performance_df['test_size'].unique()
  train_sizes = performance_df['train_size'].unique()
  fig, ax = plt.subplots(len(train_sizes), len(test_sizes), figsize=2*np.array([12, 8]))
  print(ax)
  for i, train_size in enumerate(train_sizes):
    for j, test_size in enumerate(test_sizes):
      print(i, j)
      sub_df = performance_df[(performance_df['test_size'] == test_size) & (performance_df['train_size'] == train_size)]
      predicted = sub_df['drug_prediction']
      actuals = sub_df['drug_actual']
      cm = confusion_matrix(actuals, predicted, labels=DRUGS)
      g = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DRUGS, yticklabels=DRUGS, ax=ax[i, j] if False else ax)
      g.set(title=f'Train Size: {train_size}, Test Size: {test_size}', xlabel='Predicted Drug', ylabel='Actual Drug')
  return fig

from dataclasses import dataclass

import numpy as np

def sample_mean_cov(df: pd.DataFrame):
    """
    Returns μ (1-D array of length N) and Σ (N×N array)
    """
    X = df.values.astype(float)             # M × N
    μ = X.mean(axis=0)                      # length-N
    Xc = X - μ                              # centre
    Σ = (Xc.T @ Xc) / (X.shape[0] - 1)      # unbiased sample covariance
    return μ, Σ


def greedy_mmse_columns_safe(G, Sigma, k, ridge=1e-9, tol=1e-12, max_tries=None):
    """
    Select k indices that minimize Tr(Sigma - Sigma[:,S] Sigma[S,S]^{-1} Sigma[S,:]).
    Robust Schur-complement updates + optional ridge regularization.

    Parameters
    ----------
    Sigma : (N,N) ndarray, symmetric PSD sample covariance
    k     : target subset size
    ridge : small diagonal jitter (helps with 0-variance / collinearity)
    tol   : reject candidate if Schur complement s <= tol
    max_tries : optional cap on how many candidates to scan per step (speed)

    Returns
    -------
    S : list of selected indices
    """
    N = Sigma.shape[0]
    Σ = Sigma.copy().astype(float)
    if ridge and ridge > 0:
        Σ = Σ + ridge * np.eye(N)

    def get_S(m, seed=None):
      S = []
      Ainv = None                   # will track (Σ_SS)^{-1}
      selected_mask = np.zeros(N, dtype=bool)

      # Precompute to speed marginal gain
      # We'll maintain projection residual implicitly via q = Σ[:,j] - Σ[:,S] (Ainv Σ[S,j])
      Σ_cols = [Σ[:, j] for j in range(N)]  # caching column views (optional)

      for it in range(m):
          best_j, best_gain, best_bits = None, -np.inf, None

          # Optionally restrict how many candidates you evaluate each round
          cand = np.flatnonzero(~selected_mask)
          if max_tries is not None and max_tries < cand.size:
              # simple random subset of candidates (speeds up on huge N)
              cand = np.random.choice(cand, size=max_tries, replace=False)

          if not S:
              if seed is None:
                # First pick: gain = ||Σ[:,j]||^2 / Σ[j,j]
                for j in cand:
                    c = Σ[j, j]
                    if c <= tol:      # skip zero-variance columns
                        continue
                    q = Σ_cols[j]     # since S is empty
                    gain = float((q @ q) / c)
                    if gain > best_gain:
                        best_gain, best_j = gain, j
                if best_j is None:
                    raise RuntimeError("No feasible first column found; all variances ~0?")
              else:
                 best_j = seed
              # initialize inverse as 1/c
              Ainv = np.array([[1.0 / Σ[best_j, best_j]]])
              S.append(best_j)
              selected_mask[best_j] = True
              continue

          # General step
          # We'll re-use Ainv (|S|x|S|), and for each j compute:
          # b = Σ[S, j], v = Ainv @ b, s = c - b^T v, q = Σ[:,j] - Σ[:,S] v, gain = (q^T q)/s
          Σ_S = Σ[:, S]                 # N x |S|
          Ainv_b_cache = {}

          for j in cand:
              b = Σ[S, j]               # |S|
              v = Ainv_b_cache.get(j)
              if v is None:
                  v = Ainv @ b          # |S|
                  Ainv_b_cache[j] = v
              c = Σ[j, j]
              s = c - b @ v             # Schur complement (scalar)
              if s <= tol:
                  continue               # nearly dependent → skip
              q = Σ_cols[j] - Σ_S @ v   # N-vector
              gain = float((q @ q) / s)
              if gain > best_gain:
                  best_gain, best_j, best_bits = gain, j, (b, v, s, q)

          if best_j is None:
              # Couldn’t find a numerically safe candidate; stop early
              break

          # Update Ainv via block inverse:
          # A_new = [[A, b], [b^T, c]], s = c - b^T A^{-1} b
          b, v, s, _ = best_bits
          # A_new^{-1} =
          # [Ainv + (Ainv b b^T Ainv)/s,   -Ainv b / s;
          #  (-Ainv b)^T / s,               1/s]
          Ainv_b = v                    # = Ainv @ b
          top_left = Ainv + np.outer(Ainv_b, Ainv_b) / s
          top_right = -Ainv_b[:, None] / s
          bot_left = top_right.T
          bot_right = np.array([[1.0 / s]])
          Ainv = np.block([[top_left, top_right],
                          [bot_left, bot_right]])

          S.append(best_j)
          selected_mask[best_j] = True

      return S
    
    # best_j = get_S(m=8)[-1]
    # return get_S(m=k, seed=best_j)
    return get_S(m=k, seed=None)


@dataclass(frozen=True)
class Result:
  num_features: int
  accuracy: float
  features: List[str]
  original_network_idx: int

def get_accuracies(particular_tpl):
  particular_states_df, particular_network_df = particular_tpl
  original_network_idxs = particular_states_df['original_network_idx'].unique().tolist()
  assert len(original_network_idxs) == 1
  original_network_idx = original_network_idxs[0]
  # build network
  G = nx.DiGraph()
  G.add_nodes_from(range(len(DEPENDENT_VARIABLES)))
  for _, row in particular_network_df.iterrows():
    G.add_edge(int(row['source']), int(row['target']), weight=row['weight'])

  # def get_dep_vars_drastic():
  #   dep_vars = []
  #   H = G.copy()
  #   for _ in range(N):
  #     top_node = max(H, key=lambda u: H.out_degree(u))
  #     dep_vars.append(f'node-{top_node}')
  #     for (_, v) in H.out_edges(top_node):
  #       H.remove_edge(v)
  #   return dep_vars

  def get_dep_vars():
    # only look at states of the unmodified network
    df = particular_states_df[particular_states_df['Drug'] == 'control'].drop(columns=['Drug'])
    # es_df = states_df[states_df['original_network_idx'] == original_network_idx].drop(columns=['original_network_idx', "initial_condition_idx"])
    μ, Σ = sample_mean_cov(df)   # your function from earlier
    k = 64
    idx = greedy_mmse_columns_safe(G, Σ, k, ridge=1e-8, tol=1e-12)
    selected_cols = df.columns[idx]
    return selected_cols.tolist()



  # def get_dep_vars():
  #   seen = set()
  #   dep_vars = []
  #   for _ in range(N):
  #     # Find node with highest out-degree. If ties, pick randomly.
  #     top_node = max(
  #       G,
  #       key=lambda u: (sum(
  #           bool(v not in seen)
  #           for (_, v) in G.out_edges(u)
  #         ),
  #         random.random(),
  #       ),
  #     )
  #     dep_vars.append(f'node-{top_node}')
  #     for (_, v) in G.out_edges(top_node):
  #       seen.add(v)
  #   return dep_vars

  particular_states_df = particular_states_df.drop(columns=['original_network_idx', "initial_condition_idx"])
  return list(itertools.chain(
    *(
      (
        Result(
          features=dep_vars[:top_k],
          original_network_idx=original_network_idx,
          num_features=top_k,
          accuracy=train_and_test(
            particular_states_df,
            num_trials=1,
            dep_vars=dep_vars[:top_k],
          )[0]['correct'].mean(),
        )
        for top_k in [
          int(x)
          for x in 2 ** np.arange(0, int(np.log2(N+1)))
        ] + [N]
      )
      for _ in range(1)
      if (dep_vars := get_dep_vars())
    )
  ))

if __name__ == '__main__':
  states_df = pd.read_csv(DATA_FILE, index_col=0)
  states_df = states_df.reset_index().rename(columns={"drug_name": "Drug"})

  network_df = pd.read_csv(NETWORKS_FILE, index_col=0)

  grps = {}
  states_grps = dict(tuple(states_df.groupby('original_network_idx')))
  network_grps = dict(tuple(network_df.groupby('original_network_idx')))
  for original_network_idx in states_grps:
    grps[original_network_idx] = (states_grps[original_network_idx], network_grps[original_network_idx])

  with multiprocessing.Pool() as pool:
    accuracy_data = list(itertools.chain(*tqdm.tqdm(pool.imap(
        get_accuracies,
        (
          (particular_states_df, particular_network_df)
          for particular_states_df, particular_network_df in grps.values()
        )
      ),
      total=50,
    ),
  ))

  accuracy_df = pd.DataFrame(data=[
      (result.original_network_idx, result.num_features, result.accuracy, result.features)
      for result in accuracy_data
    ],
    columns=['original_network_idx', 'num_features', 'accuracy', 'features'],
  )
  accuracy_df.to_csv('data/random-forests/dynamics-more-info-red-retention-vs-accuracy-v2-50.csv', index=False)