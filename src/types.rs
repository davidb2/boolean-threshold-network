use rand::rngs::StdRng;
use chrono::{DateTime, Utc};
use sprs::CsMat;
pub type State = Vec<bool>;
pub type Trajectories = Vec<Vec<State>>;

pub struct MetaData {
  pub tag: String,
  pub start_time: DateTime<Utc>,
  pub end_time: DateTime<Utc>,
}

pub struct BooleanThresholdNetwork {
  pub N: usize,
  pub rng: StdRng,
  /// CSC matrix.
  pub out_weights: CsMat<f64>,
  pub thresholds: Vec<f64>,
}

pub struct DynamicsConfig {
  pub num_trials: usize,
  pub num_steps: usize,
  pub seed: usize,
}

pub struct NetworkConfig {
  pub N: usize,
  pub gamma: f64,
  pub K: f64,
  pub seed: usize,
}

pub struct Edge {
  pub from: usize,
  pub to: usize,
  pub weight: f64,
}
