use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::IndexedRandom;
use crate::types::{DrugConfig, EdgePerturbation, EdgePerturbations, Network};
use crate::utils::sample_nodes;

fn get_delta<R: Rng>(drug_config: &DrugConfig, rng: &mut R) -> f64 {
  drug_config.drug_strength * vec![-1., 1.].choose(rng).unwrap()
}

pub fn get_edge_perturbations(network: &Network, drug_config: &DrugConfig) -> EdgePerturbations {
  let mut rng: StdRng = StdRng::seed_from_u64(drug_config.seed as u64);
  let target_nodes = sample_nodes(network.N, drug_config.num_targets_per_drug, &mut rng);

  assert!(network.out_weights.is_csc());

  let mut edge_perturbations = EdgePerturbations::new();
  for u in target_nodes {
    for v in 0..network.out_weights.cols() {
      // We don't need the actual weight now, just the fact that there is an edge is enough info.
      if let Some(_) = network.out_weights.get(u, v) {
        edge_perturbations.push(EdgePerturbation {
          source: u,
          target: v,
          delta: get_delta(drug_config, &mut rng),
        });
      }
    }
  }

  edge_perturbations
}