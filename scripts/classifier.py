import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from typing import *

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s :: [%(levelname)-8s] :: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

TRAIN_SIZES = [50]
TEST_SIZES = TRAIN_SIZES
NUM_PREDICTIONS = 100


@dataclass(frozen=True)
class Result:
  max_num_features: int
  accuracy: float
  original_network_idx: Optional[int] = None
  features: Optional[Set[str]] = None

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

def get_performance_data(
  df: pd.DataFrame,
  trial: int,
  train_size: int,
  test_size: int,
  *,
  dep_vars: List[str],
  original_network_idx: int,
):
  performance_data = []
  dependent_variables = dep_vars 
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
      logger.info(f'DONE {original_network_idx=} {trial=} {train_size=} {test_size=} {drug_actual=} accuracy={accuracy * 100:.2f}')
  return performance_data, clf



def train_and_test(df: pd.DataFrame, num_trials: int, original_network_idx: int, dep_vars: Optional[List[str]] = None):
  performance_data, clfs = [], []
  for data, clf in (
      get_performance_data(df, trial, train_size, test_size, dep_vars=dep_vars, original_network_idx=original_network_idx)
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