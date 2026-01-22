import argparse
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
from typing import *
from multiprocessing.pool import Pool

from classifier import train_and_test, Result

N = 5000
MUTATION_PROBABILITY = .05
NUM_GENERATIONS = 20
POPULATION_SIZE = 64
# DATA_FILE: Final[str] = 'data/drug-v1211d-N500/derived/states-1765502180766.csv'
# DATA_FILE: Final[str] = 'data/drug-v1211d-N500-2drugs/derived/states-1765513688975.csv'
DATA_FILE: Final[str] = 'data/drug-power-law-phase-transition-max-drug-strength/derived/states-1752795739394.csv'

# We use an asexual x-ploid version of the Wright-Fisher process
# (instead of the Moran process) to take advantage of parallel fitness/accuracy evaluation.

@dataclass(frozen=True)
class Individual:
  # particular_states_df: pd.DataFrame
  # original_network_idx: int
  # max_num_features: int
  features: Set[str]
  accuracy: float 


def get_score(
  features: Set[str],
  original_network_idx: int,
  particular_states_df: pd.DataFrame,
) -> float:
  return train_and_test(
    particular_states_df,
    num_trials=1,
    original_network_idx=original_network_idx,
    dep_vars=list(features),
  )[0]['correct'].mean()


class Population:
  def __init__(
    self,
    *,
    population_size: int,
    max_num_features: int,
    original_network_idx: int,
    particular_states_df: pd.DataFrame,
    pool: Pool,
  ):
    self.population_size = population_size
    self.max_num_features = max_num_features
    self.original_network_idx = original_network_idx
    self.particular_states_df = particular_states_df
    self._initialize_individuals(pool)

  def _initialize_individuals(self, pool: Pool):
    unscored_individuals = [
      Individual(
        features={f'node-{random.randint(0, N-1)}' for _ in range(self.max_num_features)},
        accuracy=0,
      )
      for _ in range(self.population_size)
    ]
    
    accuracies = pool.starmap(
      get_score, (
        (individual.features, self.original_network_idx, self.particular_states_df)
        for individual in unscored_individuals
      ),
    )

    self.individuals = [
      Individual(
        features=individual.features,
        accuracy=accuracy,
      )
      for individual, accuracy in zip(
        unscored_individuals,
        accuracies,
      )
    ]

  def next_generation(self, pool: Pool):
    self.individuals = pool.map(self._breed_individual, range(self.population_size))


  def _get_feature_from_population(self):
    parent, = random.choices(
      population=self.individuals,
      weights=np.exp(np.array([individual.accuracy for individual in self.individuals])),
    )
    return (
      f'node-{random.randint(0, N-1)}'
      if random.random() < MUTATION_PROBABILITY
      else random.choice(list(parent.features))
    )

  def _breed_individual(self, *args, **kwargs):
    features = {
      self._get_feature_from_population()
      for _ in range(self.max_num_features)
    }

    return Individual(
      features=features,
      accuracy=get_score(
        features=features,
        original_network_idx=self.original_network_idx,
        particular_states_df=self.particular_states_df,
      ),
    )


def get_accuracies(particular_states_df: pd.DataFrame, original_network_idx: int, pool: Pool):
  particular_states_df = particular_states_df.drop(
    columns=['original_network_idx', 'initial_condition_idx'],
  )

  results: List[Result] = []
  for max_num_features in [1,2,4,8,16,32,64,128]:
    population = Population(
      population_size=POPULATION_SIZE,
      max_num_features=max_num_features,
      original_network_idx=original_network_idx,
      particular_states_df=particular_states_df,
      pool=pool,
    )

    for generation_num in range(NUM_GENERATIONS):
      print('evolving generation number', generation_num)
      print(f'k={max_num_features}')
      print('best accuracy so far', max(population.individuals, key=lambda individual: individual.accuracy).accuracy)
      print('average accuracy so far', np.mean(np.array([individual.accuracy for individual in population.individuals])))
      print('----------------------------------', flush=True)
      population.next_generation(pool)

    best_individual = max(population.individuals, key=lambda individual: individual.accuracy)
    result = Result(
      original_network_idx=original_network_idx,
      max_num_features=max_num_features,
      accuracy=best_individual.accuracy,
      features=best_individual.features,
    )
    results.append(result)

  return results

def main(args: argparse.Namespace):
  states_df = pd.read_csv(DATA_FILE, index_col=0)
  states_df = states_df.reset_index().rename(columns={"drug_name": "Drug"})

  with multiprocessing.Pool() as pool:
    accuracy_data = get_accuracies(
      original_network_idx=args.original_network_idx,
      particular_states_df=states_df[states_df['original_network_idx'] == args.original_network_idx],
      pool=pool,
    )

  accuracy_df = pd.DataFrame(data=[
      (result.original_network_idx, result.max_num_features, result.accuracy, result.features)
      for result in accuracy_data
    ],
    columns=['original_network_idx', 'max_num_features', 'accuracy', 'features'],
  )
  accuracy_df.to_csv(f'data/random-forests/retention-vs-accuracy-v1217d-N5k-10drugs-genetic-shard/{args.original_network_idx}.csv', index=False)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--original-network-idx', type=int, required=True)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  main(parse_args())