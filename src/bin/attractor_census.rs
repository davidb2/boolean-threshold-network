//! Attractor census: cycle detection over many random initial conditions.
//!
//! For each network (and optionally each drug perturbation, reusing the exact
//! seeding conventions of perform_experiment), runs `--num-ics` uniformly
//! random initial conditions until a state recurrence is detected or
//! `--max-steps` is reached, and records the transient length, cycle length,
//! and a canonical attractor identity per initial condition. For every
//! distinct attractor it also writes a per-node fingerprint, the fraction of
//! the cycle each node spends in state 1, quantized to u8.
//!
//! Outputs in --output-directory:
//!   ics-<tag>.csv          per initial condition records
//!   attractors-<tag>.csv   per attractor records (row order indexes the bin)
//!   fingerprints-<tag>.bin raw u8 on-fractions, N bytes per attractor row
//!
//! Analysis and cross-perturbation comparisons happen in Python.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Mutex;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Bernoulli;
use rayon::prelude::*;

use boolean_threshold_network::drug::get_edge_perturbations;
use boolean_threshold_network::types::{
  DegreeDistribution, DrugConfig, EdgePerturbationLookup, Network, NetworkConfig,
  OutDegreeDistributionType, State,
};
use boolean_threshold_network::utils::find_gamma;

#[derive(Parser, Debug)]
struct Args {
  /// the number of networks to use
  #[arg(long)]
  num_networks: usize,

  /// the number of nodes in each network
  #[arg(short, long)]
  network_size: usize,

  /// the expected degree of a node
  #[arg(short = 'k', long)]
  expected_connectivity: Option<f64>,

  /// the value gamma in P_out(k) ~ k^-gamma
  #[arg(long)]
  gamma: Option<f64>,

  /// the distribution of the out degrees
  #[arg(long, value_enum)]
  out_degree_distribution: OutDegreeDistributionType,

  /// base seed to use for creating networks
  #[arg(long)]
  network_seed: usize,

  /// number of random initial conditions per network and perturbation
  #[arg(long)]
  num_ics: usize,

  /// base seed for initial conditions
  #[arg(long)]
  dynamics_seed: usize,

  /// stop an initial condition after this many steps without recurrence
  #[arg(long, default_value_t = 20000)]
  max_steps: usize,

  /// number of drugs; 0 runs the control only
  #[arg(long, default_value_t = 0)]
  num_drugs: usize,

  /// the number of target nodes per drug
  #[arg(long, default_value_t = 0)]
  num_targets_per_drug: usize,

  /// how strong the perturbations are
  #[arg(short = 'c', long, default_value_t = 1.0)]
  drug_strength: f64,

  /// the base seed to use for the drugs
  #[arg(long, default_value_t = 0)]
  drug_seed: usize,

  /// a tag for the output filenames
  #[arg(long)]
  tag: String,

  /// where to write the outputs
  #[arg(long)]
  output_directory: String,
}

fn pack_hash(state: &State) -> u64 {
  let mut words: Vec<u64> = vec![0; (state.len() + 63) / 64];
  for (i, &b) in state.iter().enumerate() {
    if b {
      words[i / 64] |= 1u64 << (i % 64);
    }
  }
  let mut h = DefaultHasher::new();
  words.hash(&mut h);
  h.finish()
}

fn random_state(n: usize, seed: u64) -> State {
  let mut rng = StdRng::seed_from_u64(seed);
  let coin = Bernoulli::new(0.5).unwrap();
  (0..n).map(|_| rng.sample(&coin)).collect()
}

struct IcRecord {
  network_idx: usize,
  perturbation: String,
  ic_idx: usize,
  transient: usize,
  period: usize,
  converged: bool,
  attractor_key: u64,
}

struct AttractorRecord {
  network_idx: usize,
  perturbation: String,
  attractor_key: u64,
  period: usize,
  fingerprint: Vec<u8>,
}

fn census_one_ic(
  network: &Network,
  lookup: &EdgePerturbationLookup,
  n: usize,
  ic_seed: u64,
  max_steps: usize,
) -> (usize, usize, bool, u64, Option<Vec<u8>>) {
  let mut state = random_state(n, ic_seed);
  let mut seen: HashMap<u64, usize> = HashMap::new();
  seen.insert(pack_hash(&state), 0);

  let mut transient = 0;
  let mut period = 0;
  let mut converged = false;
  for step in 1..=max_steps {
    state = network.get_next_state(&state, lookup);
    let h = pack_hash(&state);
    if let Some(&first) = seen.get(&h) {
      transient = first;
      period = step - first;
      converged = true;
      break;
    }
    seen.insert(h, step);
  }
  if !converged {
    return (max_steps, 0, false, 0, None);
  }

  // walk one full cycle to build the canonical key and the fingerprint
  let mut on_counts: Vec<u32> = vec![0; n];
  let mut key = u64::MAX;
  let cycle_start = state.clone();
  let mut s = cycle_start.clone();
  for _ in 0..period {
    let h = pack_hash(&s);
    if h < key {
      key = h;
    }
    for (j, &b) in s.iter().enumerate() {
      if b {
        on_counts[j] += 1;
      }
    }
    s = network.get_next_state(&s, lookup);
  }
  let fingerprint: Vec<u8> = on_counts
    .iter()
    .map(|&c| ((c as f64 / period as f64) * 255.0).round() as u8)
    .collect();
  (transient, period, true, key, Some(fingerprint))
}

fn main() {
  let args = Args::parse();
  std::fs::create_dir_all(&args.output_directory).expect("cannot create output dir");

  let gamma_or_k = match (args.gamma, args.expected_connectivity) {
    (Some(g), None) => g,
    (None, Some(k)) => find_gamma(k, args.network_size),
    _ => panic!("provide exactly one of --gamma or --expected-connectivity"),
  };
  let degree_distribution = match args.out_degree_distribution {
    OutDegreeDistributionType::PowerLaw => DegreeDistribution::PowerLaw { gamma: gamma_or_k },
    OutDegreeDistributionType::Homogeneous => DegreeDistribution::Homogeneous { lambda: gamma_or_k },
  };

  let ic_rows: Mutex<Vec<IcRecord>> = Mutex::new(Vec::new());
  let att_rows: Mutex<Vec<AttractorRecord>> = Mutex::new(Vec::new());

  (0..args.num_networks).into_par_iter().for_each(|network_idx| {
    let network_config = NetworkConfig {
      N: args.network_size,
      K: 0.0,
      out_degree_distribution: degree_distribution,
      reversed_edges: false,
      seed: args.network_seed + network_idx,
    };
    let network = Network::new(&network_config);

    for drug_idx in 0..=args.num_drugs {
      let perturbation_name = match drug_idx {
        0 => "control".to_string(),
        _ => format!("drug-{drug_idx}"),
      };
      let lookup: EdgePerturbationLookup = if drug_idx == 0 {
        EdgePerturbationLookup::new()
      } else {
        let drug_config = DrugConfig {
          num_drugs: args.num_drugs,
          num_targets_per_drug: args.num_targets_per_drug,
          drug_strength: args.drug_strength,
          seed: args.drug_seed + (args.num_drugs * network_idx + drug_idx),
        };
        get_edge_perturbations(&network, &drug_config)
          .iter()
          .map(|ep| ((ep.source, ep.target), ep.delta))
          .collect()
      };

      let mut local_attractors: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
      let mut local_ics: Vec<IcRecord> = Vec::with_capacity(args.num_ics);
      for ic_idx in 0..args.num_ics {
        let ic_seed = (args.dynamics_seed
          + args.num_ics * network_idx
          + ic_idx) as u64;
        let (transient, period, converged, key, fingerprint) =
          census_one_ic(&network, &lookup, args.network_size, ic_seed, args.max_steps);
        if let Some(fp) = fingerprint {
          local_attractors.entry(key).or_insert((period, fp));
        }
        local_ics.push(IcRecord {
          network_idx,
          perturbation: perturbation_name.clone(),
          ic_idx,
          transient,
          period,
          converged,
          attractor_key: key,
        });
      }
      ic_rows.lock().unwrap().extend(local_ics);
      let mut att = att_rows.lock().unwrap();
      for (key, (period, fp)) in local_attractors {
        att.push(AttractorRecord {
          network_idx,
          perturbation: perturbation_name.clone(),
          attractor_key: key,
          period,
          fingerprint: fp,
        });
      }
    }
    eprintln!("network {network_idx} done");
  });

  let ics = ic_rows.into_inner().unwrap();
  let atts = att_rows.into_inner().unwrap();

  let mut ic_file = File::create(format!(
    "{}/ics-{}.csv", args.output_directory, args.tag
  )).expect("cannot create ics csv");
  writeln!(ic_file, "network_idx,perturbation,ic_idx,transient,period,converged,attractor_key").unwrap();
  for r in &ics {
    writeln!(
      ic_file, "{},{},{},{},{},{},{}",
      r.network_idx, r.perturbation, r.ic_idx, r.transient, r.period, r.converged, r.attractor_key
    ).unwrap();
  }

  let mut att_file = File::create(format!(
    "{}/attractors-{}.csv", args.output_directory, args.tag
  )).expect("cannot create attractors csv");
  let mut fp_file = File::create(format!(
    "{}/fingerprints-{}.bin", args.output_directory, args.tag
  )).expect("cannot create fingerprints bin");
  writeln!(att_file, "network_idx,perturbation,attractor_key,period").unwrap();
  for a in &atts {
    writeln!(
      att_file, "{},{},{},{}",
      a.network_idx, a.perturbation, a.attractor_key, a.period
    ).unwrap();
    fp_file.write_all(&a.fingerprint).unwrap();
  }

  println!(
    "wrote {} ic records, {} attractors ({} converged ICs)",
    ics.len(), atts.len(), ics.iter().filter(|r| r.converged).count()
  );
}
