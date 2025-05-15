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