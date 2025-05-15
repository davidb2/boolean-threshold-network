// src/lib/network.rs
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Uniform};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use sprs::{TriMat};

use crate::types::{BooleanThresholdNetwork, NetworkConfig, State, Edge};

fn uniform_weight<R: Rng>(rng: &mut R) -> f64 {
  let dist = Uniform::new(-1., 1.).unwrap();
  rng.sample(dist)
}

impl BooleanThresholdNetwork {
  /// Build a new random network from the given config.
  pub fn new(config: &NetworkConfig) -> Self {
    let mut net = BooleanThresholdNetwork {
      N: config.N,
      rng: StdRng::seed_from_u64(config.seed as u64),
      out_weights: TriMat::new((config.N, config.N)).to_csr(),
      thresholds: vec![0.0; config.N],
    };
    net.generate_weights(config);
    net.generate_thresholds(config);
    net
  }

  /// Sample a uniformly random Boolean state of length N.
  pub fn get_uniformly_random_state<R: Rng>(&self, trial_rng: &mut R) -> State {
    // Use the trial_rng for reproducibility purposes.
    let coin = Bernoulli::new(0.5).unwrap();
    (0..self.N)
      .map(|_| trial_rng.sample(&coin))
      .collect::<Vec<bool>>()
  }

  /// Compute the next state given the previous.
  pub fn get_next_state(&self, previous: &State) -> State {
    assert_eq!(previous.len(), self.N);
    let mut next = vec![false; self.N];
    // For each target node j, sum up incoming weights * state[i]
    for j in 0..self.N {
      let mut sum = 0.0;
      // iterate over nonzeros in column j
      // sprs stores CSR by rows, so we can transpose or iterate rows of transposed:
      for (i, &w) in self
        .out_weights
        .outer_view(j).expect("invalid CSC")
        .iter()
      {
        let b: f64 = previous[i].into();
        sum += w * b;
      }
      next[j] = activation(sum, self.thresholds[j], previous[j]);
    }
    next
  }

  /// Returns the threshold vector.
  pub fn get_thresholds(&self) -> &[f64] {
    &self.thresholds
  }

  /// Flatten out into a list of WeightedEdges (from, to, weight).
  pub fn get_representation_of_network(&self) -> Vec<Edge> {
    let mut edges = Vec::new();
    for (&w, (row, col)) in self.out_weights.iter() {
      edges.push(Edge {
        from: row,
        to: col,
        weight: w,
      });
    }
    edges
  }

  // — internal helpers —

  fn generate_thresholds(&mut self, _params: &NetworkConfig) {
    // currently all zeros; override if you want randomness
    self.thresholds.fill(0.0);
  }

  fn generate_weights(&mut self, params: &NetworkConfig) {
    // 1) sample out‐degrees
    let out_degrees = self.generate_out_degree_distribution(params);

    // 2) build triplets
    let mut tri = TriMat::with_capacity(
      (self.N, self.N),
      out_degrees.iter().sum::<usize>()
    );
    for u in 0..self.N {
      let neighbors = sample_neighbors(self.N, out_degrees[u], &mut self.rng);
      for &v in &neighbors {
        let w = uniform_weight(&mut self.rng);
        tri.add_triplet(u, v, w);
      }
    }

    // 3) finalize CSR
    self.out_weights = tri.to_csc();
  }


  fn generate_out_degree_distribution(&mut self, params: &NetworkConfig) -> Vec<usize> {
    // P(k) ∝ k^{-γ}, for k = 1..N
    let weights: Vec<f64> = (1..=self.N)
      .map(|k| (k as f64).powf(-params.gamma))
      .collect();
    let dist = WeightedIndex::new(&weights).unwrap();

    // sample one per node
    (0..self.N)
      .map(|_| dist.sample(&mut self.rng) + 1)  // +1 because sample returns 0-based
      .collect()
  }
} // end impl BooleanThresholdNetwork

/// Activation: if sum == threshold, stay same; else true iff sum > threshold.
fn activation(sum: f64, threshold: f64, prev: bool) -> bool {
  if (sum - threshold).abs() < f64::EPSILON {
    prev
  } else {
    sum > threshold
  }
}

/// Sample `k` unique neighbors in `[0, N)` in expectation O(k log k) for k << N.
fn sample_neighbors<R: Rng>(
  N: usize,
  k: usize,
  rng: &mut R,
) -> Vec<usize> {
  use std::collections::HashSet;
  let mut seen = HashSet::with_capacity(k);
  let die = Uniform::new(0, N).unwrap();
  while seen.len() < k {
    seen.insert(rng.sample(die));
  }
  seen.into_iter().collect()
}
