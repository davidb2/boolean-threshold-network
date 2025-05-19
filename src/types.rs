use std::collections::HashMap;
use chrono::{DateTime, Utc};
use clap::ValueEnum;
use sprs::CsMat;

pub type State = Vec<bool>;
pub type Trajectories = Vec<Trajectory>;
pub type States = Vec<State>;
pub type Perturbations = Vec<Perturbation>;
pub type EdgePerturbations = Vec<EdgePerturbation>;
pub type ExperimentResults = Vec<ExperimentResult>;
pub type EdgeIndex = (usize, usize);
pub type EdgePerturbationLookup = HashMap<EdgeIndex, f64>;

#[derive(ValueEnum, Debug, Clone)]
pub enum OutDegreeDistributionType {
  Homogeneous,
  PowerLaw,
}

#[derive(Debug, Copy, Clone)]
pub enum DegreeDistribution {
  /// Poisson distribution.
  Homogeneous {
    lambda: f64,
  },
  /// Power law / scale-free distribution.
  PowerLaw {
    gamma: f64,
  },
}
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
  pub K: f64,
  pub out_degree_distribution: DegreeDistribution,
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
  pub edge_perturbations: EdgePerturbations,
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

pub struct EdgePerturbation {
  pub source: usize,
  pub target: usize,
  /// new weight <- old_weight + delta
  /// so, old_weight <- new_weight - delta
  pub delta: f64,
}
