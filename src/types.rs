use rand::rngs::StdRng;
use chrono::{DateTime, Utc};
use sprs::CsMat;

pub type State = Vec<bool>;
pub type Trajectories = Vec<Trajectory>;
pub type States = Vec<State>;
pub type Perturbations = Vec<Perturbation>;
pub type ExperimentResults = Vec<ExperimentResult>;

pub struct ExperimentConfig {
  pub num_networks: usize,
  pub drug_config: DrugConfig,
  pub network_config: NetworkConfig,
  pub dynamics_config: DynamicsConfig,
}

pub struct MetaData {
  pub tag: String,
  pub start_time: DateTime<Utc>,
  pub end_time: Option<DateTime<Utc>>,
}

pub struct DynamicsConfig {
  pub num_initial_conditions: usize,
  pub num_steps: usize,
  pub seed: usize,
}

pub struct NetworkConfig {
  pub N: usize,
  pub gamma: f64,
  pub K: f64,
  pub seed: usize,
}

pub struct DrugConfig {
  pub num_drugs: usize,
  pub num_targets_per_drug: usize,
  pub drug_strength: f64,
  pub seed: usize,
}

pub struct ExperimentResult {
  pub original_network: Network,
  pub perturbations: Perturbations,
  pub network_idx: usize,
}

pub struct Perturbation {
  pub name: String,
  pub perturbed_network: Network,
  pub trajectories: Trajectories,
}

pub struct Trajectory {
  pub states: States,
  pub initial_condition_idx: usize,
}

#[derive(Clone)]
pub struct Network {
  pub N: usize,
  // pub rng: StdRng,
  /// CSC matrix.
  pub out_weights: CsMat<f64>,
}

pub struct Edge {
  pub source: usize,
  pub target: usize,
  pub weight: f64,
}
