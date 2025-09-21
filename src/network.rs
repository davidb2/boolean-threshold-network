// src/lib/network.rs
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Poisson, Uniform};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use sprs::{TriMat};

use crate::types::{Network, NetworkConfig, State, Edge, DynamicsConfig, EdgePerturbationLookup, DegreeDistribution};
use crate::utils::sample_nodes;

fn uniform_weight<R: Rng>(rng: &mut R) -> f64 {
  let dist = Uniform::new(-1., 1.).unwrap();
  rng.sample(dist)
}

impl Network {
  /// Build a new random network from the given config.
  pub fn new(config: &NetworkConfig) -> Self {
    let mut net = Network {
      N: config.N,
      out_weights: TriMat::new((config.N, config.N)).to_csr(),
    };

    let mut rng = StdRng::seed_from_u64(config.seed as u64);
    net.generate_weights(config, &mut rng);
    assert!(net.out_weights.is_csc());
    net
  }

  /// Compute the next state given the previous.
  pub fn get_next_state(
    &self,
    previous: &State,
    edge_perturbation_lookup: &EdgePerturbationLookup,
  ) -> State {
    assert_eq!(previous.len(), self.N);
    assert!(self.out_weights.is_csc());
    let mut next = vec![false; self.N];
    // For each target node j, sum up incoming weights * state[i]
    for j in 0..self.N {
      let mut sum = 0.0;
      // iterate over nonzeros in column j
      // sprs stores CSR by rows, so we can transpose or iterate rows of transposed:
      for (i, &weight) in self
        .out_weights
        .outer_view(j).expect("invalid CSC")
        .iter()
      {
        if previous[i] {
          let perturbed_weight = weight + match edge_perturbation_lookup.get(&(i, j)) {
            None => 0.,
            Some(delta) => *delta,
          };
          sum += perturbed_weight;
        }
      }
      next[j] = activation(sum, previous[j]);
    }
    next
  }

  /// Flatten out into a list of WeightedEdges (from, to, weight).
  pub fn get_representation_of_network(&self) -> Vec<Edge> {
    let mut edges = Vec::new();
    for (&w, (row, col)) in self.out_weights.iter() {
      edges.push(Edge {
        source: row,
        target: col,
        weight: w,
      });
    }
    edges
  }

  fn generate_weights<R: Rng>(&mut self, params: &NetworkConfig, rng: &mut R) {
    assert!(self.out_weights.is_csr());
    // 1) sample out‐degrees
    let out_degrees = generate_out_degree_distribution(self, params, rng);

    // 2) build triplets
    let mut tri = TriMat::with_capacity(
      (self.N, self.N),
      out_degrees.iter().sum::<usize>()
    );
    for u in 0..self.N {
      let neighbors = sample_nodes(self.N, out_degrees[u], rng);
      for &v in &neighbors {
        let w = uniform_weight(rng);
        if params.reversed_edges {
          tri.add_triplet(v, u, w);
        } else {
          tri.add_triplet(u, v, w);
        }
      }
    }

    // 3) finalize CSR
    self.out_weights = tri.to_csc();
  }
} // end impl BooleanThresholdNetwork

/// Sample a uniformly random Boolean state of length N.
pub fn get_uniformly_random_state(network_size: usize, dynamics_config: &DynamicsConfig) -> State {
  let mut initial_condition_rng = StdRng::seed_from_u64(dynamics_config.seed as u64);
  let coin = Bernoulli::new(0.5).unwrap();
  (0..network_size)
    .map(|_| initial_condition_rng.sample(&coin))
    .collect::<Vec<bool>>()
}

fn generate_out_degree_distribution<R: Rng>(network: &Network, params: &NetworkConfig, rng: &mut R) -> Vec<usize> {
  match params.out_degree_distribution {
    DegreeDistribution::Homogeneous { lambda } => generate_out_degree_distribution_homogeneous(network, lambda, rng),
    DegreeDistribution::PowerLaw { gamma } => generate_out_degree_distribution_powerlaw(network, gamma, rng),
  }
}

fn generate_out_degree_distribution_homogeneous<R: Rng>(network: &Network, lambda: f64, rng: &mut R) -> Vec<usize> {
  if lambda == 0. {
    return (0..network.N).map(|_| 0).collect();
  }
  // P(k) ∝ \lambda^k/k!, for k = 0..N where \lambda = K
  let dist = Poisson::new(lambda).unwrap();

  // sample one per node
  (0..network.N)
    .map(|_| dist.sample(rng) as usize)
    .collect()
}

fn generate_out_degree_distribution_powerlaw<R: Rng>(network: &Network, gamma: f64, rng: &mut R) -> Vec<usize> {
  // P(k) ∝ k^{-γ}, for k = 1..N
  let weights: Vec<f64> = (1..=network.N)
    .map(|k| (k as f64).powf(-gamma))
    .collect();
  let dist = WeightedIndex::new(&weights).unwrap();

  // sample one per node
  (0..network.N)
    .map(|_| dist.sample(rng) + 1)  // +1 because sample returns 0-based
    .collect()
}

/// Activation: if sum == 0, stay same; else true iff sum > threshold.
fn activation(sum: f64, prev: bool) -> bool {
  if sum.abs() < f64::EPSILON {
    prev
  } else {
    sum > 0.
  }
}
