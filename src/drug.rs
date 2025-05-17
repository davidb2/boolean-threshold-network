use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::IndexedRandom;
use crate::types::{DrugConfig, Network};
use crate::utils::sample_nodes;

fn perturb<R: Rng>(weight: f64, drug_config: &DrugConfig, rng: &mut R) -> f64 {
  weight + drug_config.drug_strength * vec![-1., 1.].choose(rng).unwrap()
}

pub fn get_perturbed_network(network: &Network, drug_config: &DrugConfig) -> Network {
  let mut rng: StdRng = StdRng::seed_from_u64(drug_config.seed as u64);
  let target_nodes = sample_nodes(network.N, drug_config.num_targets_per_drug, &mut rng);

  let mut perturbed_network: Network = network.clone();
  assert!(perturbed_network.out_weights.is_csc());

  for u in target_nodes {
    for v in 0..perturbed_network.out_weights.cols() {
      if let Some(weight) = perturbed_network.out_weights.get_mut(u, v) {
        *weight = perturb(*weight, drug_config, &mut rng);
      }
    }
  }

  perturbed_network
}