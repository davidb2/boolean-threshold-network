import argparse
import itertools
import logging
import math
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import pathlib
import random
import seaborn as sns
import tqdm

from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from multiprocessing.pool import Pool

from classifier import train_and_test, Result

MUTATION_PROBABILITY = .05
NUM_GENERATIONS = 40
POPULATION_SIZE = 128
# DATA_FILE: Final[str] = 'data/drug-v1211d-N500/derived/states-1765502180766.csv'
# DATA_FILE: Final[str] = 'data/drug-v1211d-N500-2drugs/derived/states-1765513688975.csv'
# DATA_FILE: Final[str] = 'data/drug-power-law-phase-transition-max-drug-strength/derived/states-1752795739394.csv'
# DATA_FILE: Final[str] = 'data/drug-v0127d-N5k-gamma1.8-10drugs/derived/states-1769568379340.csv'

# We use an asexual x-ploid version of the Wright-Fisher process
# (instead of the Moran process) to take advantage of parallel fitness/accuracy evaluation.
class TypeOfReproduction(Enum):
  CROSSOVER = auto()
  COPY_WITH_MUTATION = auto()

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
    network_size: int,
    population_size: int,
    max_num_features: int,
    original_network_idx: int,
    particular_states_df: pd.DataFrame,
    pool: Pool,
  ):
    self.network_size = network_size
    self.population_size = population_size
    self.max_num_features = max_num_features
    self.original_network_idx = original_network_idx
    self.particular_states_df = particular_states_df
    self._initialize_individuals(pool)

  def _initialize_individuals(self, pool: Pool):
    unscored_individuals = [
      Individual(
        features={f'node-{random.randint(0, self.network_size-1)}' for _ in range(self.max_num_features)},
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


  def _breed_individual(self, tor: TypeOfReproduction):
    if tor == TypeOfReproduction.CROSSOVER: return self._breed_individual_crossover()
    if tor == TypeOfReproduction.COPY_WITH_MUTATION: return self._breed_individual_copy_with_mutation()
    raise ValueError(f'cannot breed individual using {tor}')


  def next_generation(self, pool: Pool):
    crossover_amount = self.population_size // 2
    type_of_reproductions = (
      [TypeOfReproduction.CROSSOVER] * crossover_amount +
      [TypeOfReproduction.COPY_WITH_MUTATION] * (self.population_size-crossover_amount)
    )

    self.individuals = pool.map(self._breed_individual, type_of_reproductions)

  def _get_individual_from_population(self):
    individual, = random.choices(
      population=self.individuals,
      weights=np.exp(np.array([individual.accuracy for individual in self.individuals])),
    )
    return individual

  def _get_feature_from_population(self):
    parent = self._get_individual_from_population()
    return (
      f'node-{random.randint(0, self.network_size-1)}'
      if random.random() < MUTATION_PROBABILITY
      else random.choice(list(parent.features))
    )

  def _breed_individual_copy_with_mutation(self, *args, **kwargs):
    parent = self._get_individual_from_population()
    features = {
      (
        self._get_feature_from_population()
        if random.random() < MUTATION_PROBABILITY
        else feature
      )
      for feature in parent.features
    }
    return Individual(
      features=features,
      accuracy=get_score(
        features=features,
        original_network_idx=self.original_network_idx,
        particular_states_df=self.particular_states_df,
      ),
    )

  def _breed_individual_crossover(self, *args, **kwargs):
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


def get_accuracies(
  particular_states_df: pd.DataFrame,
  original_network_idx: int,
  network_size: int,
  pool: Pool,
  feature_sizes: Optional[List[int]] = None,
):
  '''Generator: yields a list of rows for each feature_size after it finishes all generations.

  Each row is (original_network_idx, max_num_features, generation, best_accuracy, features).
  `features` is only populated for the final generation of each feature_size.
  Yielding after each feature_size allows the caller to write partial results incrementally.
  '''
  particular_states_df = particular_states_df.drop(
    columns=['original_network_idx', 'initial_condition_idx'],
  )

  if feature_sizes is None:
    powers = [4**i for i in range(int(math.log(network_size, 4)) + 1)]
    feature_sizes = powers if powers[-1] == network_size else powers + [network_size]

  for max_num_features in feature_sizes:
    rows = []
    population = Population(
      network_size=network_size,
      population_size=POPULATION_SIZE,
      max_num_features=max_num_features,
      original_network_idx=original_network_idx,
      particular_states_df=particular_states_df,
      pool=pool,
    )

    # generation 0: initial population before any evolution
    best = max(population.individuals, key=lambda individual: individual.accuracy)
    rows.append((original_network_idx, max_num_features, 0, best.accuracy, best.features))
    print(f'generation 0, k={max_num_features}, best={best.accuracy:.4f}', flush=True)

    for generation_num in range(1, NUM_GENERATIONS + 1):
      population.next_generation(pool)
      best = max(population.individuals, key=lambda individual: individual.accuracy)
      avg = np.mean([ind.accuracy for ind in population.individuals])
      rows.append((original_network_idx, max_num_features, generation_num, best.accuracy, best.features))
      print(f'generation {generation_num}, k={max_num_features}, best={best.accuracy:.4f}, avg={avg:.4f}', flush=True)

    yield rows


def main(args: argparse.Namespace):
  output_path = pathlib.Path(f'{args.output_dir}/{args.original_network_idx}.csv')
  done_path = pathlib.Path(f'{args.output_dir}/{args.original_network_idx}.done')
  if done_path.exists():
    return None

  states_df = pd.read_csv(args.states_file, index_col=0)
  states_df = states_df.reset_index().rename(columns={"drug_name": "Drug"})

  # remove any partial csv from a previous incomplete run
  output_path.unlink(missing_ok=True)

  cols = ['original_network_idx', 'max_num_features', 'generation', 'best_accuracy', 'features']
  first_chunk = True
  with multiprocessing.Pool() as pool:
    for rows in get_accuracies(
      original_network_idx=args.original_network_idx,
      particular_states_df=states_df[states_df['original_network_idx'] == args.original_network_idx],
      network_size=args.network_size,
      pool=pool,
      feature_sizes=args.feature_sizes,
    ):
      pd.DataFrame(rows, columns=cols).to_csv(output_path, mode='a', header=first_chunk, index=False)
      first_chunk = False

  done_path.touch()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--original-network-idx', type=int, required=True)
  parser.add_argument('--states-file', type=str, required=True)
  parser.add_argument('--output-dir', type=str, required=True)
  parser.add_argument('--network-size', type=int, default=5000)
  parser.add_argument('--feature-sizes', type=int, nargs='+', default=None,
                      help='feature set sizes to test (default: powers of 2 up to network-size)')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  main(parse_args())
