import itertools
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import random
import tqdm

from typing import *

from classifier import Result, train_and_test


N = 5_00
DEPENDENT_VARIABLES: Final[List[str]] = [f'node-{x}' for x in range(N)]
# DATA_FILE: Final[str] = 'data/drug-v1211d-N500/derived/states-1765502180766.csv'
DATA_FILE: Final[str] = 'data/drug-v1211d-N500-2drugs/derived/states-1765513688975.csv'




def get_accuracies(tpl: Tuple[pd.DataFrame, int]):
  particular_states_df, original_network_idx = tpl
  particular_states_df = particular_states_df.drop(columns=['original_network_idx', "initial_condition_idx"])
  return list(itertools.chain(
    *(
      (
        Result(num_features=top_k, accuracy=train_and_test(particular_states_df, num_trials=1, original_network_idx=original_network_idx, dep_vars=dep_vars[:top_k])[0]['correct'].mean())
        for top_k in list(range(1,15+1)) + [
          int(x) for x in 2 ** np.arange(4, int(np.log2(N+1)))
        ] + [N]
      )
      for _ in range(5)
      if (dep_vars := sorted(DEPENDENT_VARIABLES, key=lambda _: random.random()))
    )
  ))


if __name__ == '__main__':
  states_df = pd.read_csv(DATA_FILE, index_col=0)
  states_df = states_df.reset_index().rename(columns={"drug_name": "Drug"})

  with multiprocessing.Pool() as pool:
    accuracy_data = list(itertools.chain(*tqdm.tqdm(pool.imap(
        get_accuracies,
        (
          (particular_states_df, original_network_idx)
          for original_network_idx, particular_states_df in states_df.groupby('original_network_idx')
        )
      ),
      total=50,
    ),
  ))

  accuracy_df = pd.DataFrame(data=[
      (result.num_features, result.accuracy)
      for result in accuracy_data
    ],
    columns=['num_features', 'accuracy'],
  )
  accuracy_df.to_csv('data/random-forests/retention-vs-accuracy-v1211d-2drugs.csv', index=False)