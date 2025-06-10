use rand::SeedableRng;
use rayon::prelude::*;
use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::Bernoulli;
use crate::drug::get_edge_perturbations;
use crate::network::get_uniformly_random_state;
use crate::types::{
  Network,
  DynamicsConfig,
  State,
  NetworkConfig,
  DrugConfig,
  ExperimentConfig,
  ExperimentResults,
  ExperimentResult,
  Trajectory,
  Perturbation,
  EdgePerturbations,
  EdgePerturbationLookup,
};

fn flip_some_bits(state: &State, correlation: f64, seed: usize) -> State {
  assert!(0. <= correlation && correlation <= 1., "Correlation must be in [0, 1]");
  let mut flip_rng = StdRng::seed_from_u64(seed as u64);
  let biased_coin = Bernoulli::new(1.-correlation).unwrap();
  state
    .iter()
    .map(|b| match flip_rng.sample(&biased_coin) {
      false => *b,
      true => !b,
    })
    .collect()
}

fn run_for_one_initial_condition(
  initial_condition_idx: usize,
  network_idx: usize,
  original_network: &Network,
  edge_perturbation_lookup: &EdgePerturbationLookup,
  experiment_config: &ExperimentConfig
) -> Trajectory {
  let base_seed =  experiment_config.dynamics_config.seed + (
    experiment_config.dynamics_config.num_initial_conditions * network_idx
  );
  let dynamics_config = DynamicsConfig {
    seed: base_seed,
    ..experiment_config.dynamics_config
  };

  let mut state = get_uniformly_random_state(experiment_config.network_config.N, &dynamics_config);
  if initial_condition_idx > 0 {
    state = flip_some_bits(&state, experiment_config.dynamics_config.initial_condition_correlation, base_seed + initial_condition_idx);
  }

  let mut states: Vec<State> = Vec::with_capacity(experiment_config.dynamics_config.num_steps + 1);
  for step in 0..experiment_config.dynamics_config.num_steps {
    if experiment_config.dynamics_config.num_steps - step < experiment_config.dynamics_config.num_final_states_to_store {
      states.push(state.clone());
    }
    state = original_network.get_next_state(&state, edge_perturbation_lookup);
  }
  states.push(state.clone());

  Trajectory {
    initial_condition_idx,
    states,
  }
}
fn run_for_one_drug(
  drug_idx: usize,
  network_idx: usize,
  network: &Network,
  experiment_config: &ExperimentConfig
) -> Perturbation {
  let drug_config = DrugConfig {
    // Seed is unique to network and drug
    seed: experiment_config.drug_config.seed + (
      experiment_config.drug_config.num_drugs * network_idx + drug_idx
    ),
    ..experiment_config.drug_config
  };
  let edge_perturbations: EdgePerturbations = match drug_idx {
    0 => EdgePerturbations::new(),
    _ => get_edge_perturbations(&network, &drug_config),
  };

  // Convert edge perturbations to fast lookup data structure.
  let edge_perturbation_lookup: EdgePerturbationLookup = edge_perturbations
    .iter()
    .map(|edge_perturbation| {
      ((edge_perturbation.source, edge_perturbation.target), edge_perturbation.delta)
    })
    .collect();

  let trajectories =
    (0..experiment_config.dynamics_config.num_initial_conditions)
      .map(|initial_condition_idx| run_for_one_initial_condition(
        initial_condition_idx,
        network_idx,
        network,
        &edge_perturbation_lookup,
        experiment_config
      ))
      .collect();

  Perturbation {
    name: match drug_idx {
      0 => "control".to_string(),
      _ => format!("drug-{drug_idx}", drug_idx=drug_idx),
    },
    edge_perturbations,
    trajectories,
  }
}

fn run_for_one_network(
  network_idx: usize,
  experiment_config: &ExperimentConfig,
) -> ExperimentResult {
  let network_config = NetworkConfig {
    seed: experiment_config.network_config.seed + network_idx,
    ..experiment_config.network_config
  };
  let network = Network::new(&network_config);
  let perturbations =
    (0..experiment_config.drug_config.num_drugs+1) // drug_idx 0 is the control
      .map(|drug_idx| run_for_one_drug(drug_idx, network_idx, &network, experiment_config))
      .collect();

  ExperimentResult {
    original_network: network,
    perturbations,
    network_idx,
  }
}

/// Fill `trajectories` with `num_trials` runs of length `num_steps+1`.
pub fn run_experiment(
  experiment_config: &ExperimentConfig,
) -> ExperimentResults {
  // Parallelize over the various networks.
  let experiment_results =
    (0..experiment_config.num_networks)
      .into_par_iter()
      .map(|network_idx| run_for_one_network(network_idx, experiment_config))
      .collect();

  experiment_results
}
