use std::path::PathBuf;
use chrono::Utc;
use uuid::Uuid;
use clap::Parser;
use boolean_threshold_network::experiment::run_experiment;
use boolean_threshold_network::types::{DrugConfig, DynamicsConfig, ExperimentConfig, MetaData, NetworkConfig, OutDegreeDistributionType};
use boolean_threshold_network::types::DegreeDistribution::{Homogeneous, PowerLaw};
use boolean_threshold_network::utils::{
  average_connectivity,
  find_gamma,
};
use boolean_threshold_network::writer::write_protobuf;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// the number of networks to use
  #[arg(long)]
  num_networks: usize,

  /* START: Network config. */
  /// the number of nodes in each network
  #[arg(short, long = "network-size")]
  N: usize,

  /// the expected degree of a node
  #[arg(short, long = "expected-connectivity")]
  K: Option<f64>,

  /// the value \gamma in P_{out}(k) \propto k^{-\gamma}
  #[arg(long)]
  gamma: Option<f64>,

  /// the value for the poisson distribution
  #[arg(long)]
  lambda: Option<f64>,

  /// the distribution of the out degrees.
  #[arg(long, value_enum)]
  out_degree_distribution: OutDegreeDistributionType,

  /// if true, all edges will be reversed.
  #[arg(long)]
  reversed_edges: bool,

  /// base seed to use for creating networks
  #[arg(long)]
  network_seed: usize,
  /* END: Network config. */

  /* START: Dynamics config. */
  /// number of times to run the same dynamics, but for different initial conditions
  #[arg(long)]
  num_initial_conditions: usize,

  /// for each initial condition, how many iterations of the dynamics update
  #[arg(long)]
  num_steps: usize,

  /// base seed use for dynamics
  #[arg(long)]
  dynamics_seed: usize,

  /// the correlation for the initial states of each network->drug trial
  #[arg(long)]
  initial_condition_correlation: f64,

  /// number of final states to store
  #[arg(long)]
  num_final_states_to_store: usize,
  /* END: Dynamics config. */

  /* START: Drug config. */
  /// the number of different drug profiles to use; each drug is specific to a network
  #[arg(long)]
  num_drugs: usize,

  /// the number of target nodes per drug
  #[arg(long)]
  num_targets_per_drug: usize,

  /// how strong the perturbations on the network are for each drug
  #[arg(short = 'c', long)]
  drug_strength: f64,

  /// the base seed to use for the drugs
  #[arg(long)]
  drug_seed: usize,
  /* END: Drug config. */

  /* START: Metadata. */
  /// a tag to stick in the metadata
  #[arg(long)]
  tag: String,
  /* END: Metadata. */

  /// the output directory of the experiment results
  #[arg(long)]
  output_directory: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
  let args = Args::parse();

  // Validate the args.
  println!("Validating inputs...");
  assert!(0. <= args.drug_strength && args.drug_strength <= 1., "drug strength must be between 0 and 1.");
  assert!(
    0. <= args.initial_condition_correlation
    && args.initial_condition_correlation <= 1.,
    "initial condition correlation must be in [0, 1]."
  );
  assert!(
    args.num_final_states_to_store <= args.num_steps,
    "cannot store more final states than number of steps"
  );
  assert!(args.output_directory.exists(), "the provided output directory does not exist.");

  /* BEGIN: Network config. */
  let out_degree_distribution = match (args.out_degree_distribution, args.K, args.gamma, args.lambda) {
    (_, None, None, None) => panic!("need to specify one of K, lambda, or gamma."),
    (_, Some(_), Some(_), _) |
    (_, _, Some(_), Some(_)) |
    (_, Some(_), _, Some(_)) => panic!("can only specify one of K, lambda, or gamma."),
    (OutDegreeDistributionType::Homogeneous, Some(K), _, _) => {
      assert!(0. <= K && K <= (args.N as f64), "for homogeneous distributions, it must be that 0 <= K <= N");
      Homogeneous { lambda: K }
    },
    (OutDegreeDistributionType::Homogeneous, _, _, Some(lambda)) => Homogeneous { lambda },
    (OutDegreeDistributionType::PowerLaw, Some(K), _, _) => {
      assert!(1. <= K && K <= (args.N+1) as f64 / 2., "for power law distributions, it must be that 1 <= K <= (N+1)/2.");
      PowerLaw { gamma: find_gamma(K, args.N) }
    },
    (OutDegreeDistributionType::PowerLaw, _, Some(gamma), _) => PowerLaw { gamma },
    (_, _, _, _) => panic!("specified out degree distribution does not match given parameters.")
  };
  let K = match (args.K, out_degree_distribution) {
    (Some(K), _) => K,
    (_, Homogeneous { lambda}) => lambda,
    (_, PowerLaw { gamma }) => average_connectivity(gamma, args.N),
  };
  let network_config = NetworkConfig {
    N: args.N,
    K,
    out_degree_distribution,
    reversed_edges: args.reversed_edges,
    seed: args.network_seed,
  };
  /* END: Network config. */

  /* BEGIN: Dynamics config. */
  let dynamics_config = DynamicsConfig {
    num_steps: args.num_steps,
    num_initial_conditions: args.num_initial_conditions,
    initial_condition_correlation: args.initial_condition_correlation,
    num_final_states_to_store: args.num_final_states_to_store,
    seed: args.dynamics_seed,
  };
  /* END: Dynamics config. */

  /* BEGIN: Drug config. */
  let drug_config = DrugConfig {
    num_drugs: args.num_drugs,
    num_targets_per_drug: args.num_targets_per_drug,
    drug_strength: args.drug_strength,
    seed: args.drug_seed,
  };
  /* END: Drug config. */

  /* BEGIN: Metadata. */
  let mut metadata = MetaData {
    tag: args.tag,
    start_time: Utc::now(),
    end_time: None,
  };
  /* END: Metadata. */

  let output_path = args.output_directory.join(
   format!(
     "experiment-{timestamp}-{id}.pb",
     timestamp=metadata.start_time.timestamp_millis(),
     id=Uuid::new_v4().to_string(),
   )
  );
  let output_filename = output_path.to_str().unwrap();

  let experiment_config = ExperimentConfig {
    num_networks: args.num_networks,
    network_config,
    dynamics_config,
    drug_config,
  };

  println!("Running experiment...");
  let experiment_results = run_experiment(&experiment_config);
  metadata.end_time = Some(Utc::now());

  println!("Writing experiment results to file...");
  write_protobuf(
    &experiment_results,
    &experiment_config,
    &metadata,
    output_filename,
  )?;

  println!("Done.");
  Ok(())
}