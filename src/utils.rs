use rand::distr::Uniform;
use rand::Rng;
use rootfinder::{root_bisection_fast, Interval};

pub fn zeta(s: f64, N: usize) -> f64 {
  (1..=N).map(|n| f64::powf(n as f64, -s)).sum()
}

pub fn average_connectivity(gamma: f64, N: usize) -> f64 {
  zeta(gamma-1., N) / zeta(gamma, N)
}
pub fn find_gamma(K: f64, N: usize) -> f64 {
  let f = |gamma| { average_connectivity(gamma, N) - K };
  root_bisection_fast(&f, Interval::new(0., N as f64))
}

/// Sample `k` unique neighbors in `[0, N)` in expectation O(k log k) for k << N.
pub fn sample_nodes<R: Rng>(
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