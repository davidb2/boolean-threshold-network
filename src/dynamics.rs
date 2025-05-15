use rand::{SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::types::{BooleanThresholdNetwork, DynamicsConfig, Trajectories, State};

/// Fill `trajectories` with `num_trials` runs of length `num_steps+1`.
pub fn compute_trajectories(
  network: &BooleanThresholdNetwork,
  config: &DynamicsConfig,
) -> Trajectories {
  // 1) allocate outer & inner vectors
  let trials = config.num_trials;
  let steps_plus_one = config.num_steps + 1;

  // 2) parallelize over trials
  (0..trials)
    .into_par_iter()
    .map(|trial| {
      // derive a per-trial RNG seed for reproducibility
      let seed = config.seed.wrapping_add(trial) as u64;
      let mut trial_rng = StdRng::seed_from_u64(seed);

      // initial state
      let mut state = network.get_uniformly_random_state(&mut trial_rng);
      let mut trajectory: Vec<State> = Vec::with_capacity(steps_plus_one);

      // record trajectory
      for _ in 0..steps_plus_one {
        trajectory.push(state.clone());
        state = network.get_next_state(&state);
      }
      trajectory
    })
    .collect()
}
