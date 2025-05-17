use std::path::PathBuf;
use chrono::Utc;
use clap::Parser;
use boolean_threshold_network::experiment::run_experiment;
use boolean_threshold_network::types::{DrugConfig, DynamicsConfig, ExperimentConfig, MetaData, NetworkConfig};
use boolean_threshold_network::utils::{average_connectivity, find_gamma};
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
  assert!(args.drug_strength >= 0., "drug strength cannot be negative.");
  assert!(args.output_directory.exists(), "the provided output directory does not exist.");

  /* BEGIN: Network config. */
  let gamma = match args.gamma {
    Some(gamma) => {
      assert!(gamma >= 0., "gamma cannot be negative.");
      gamma
    },
    None => match args.K {
      None => panic!("only one of gamma or K should be provided."),
      Some(K) => {
        assert!(K > 0., "K needs to be positive if supplied.");
        find_gamma(K, args.N)
      },
    }
  };
  let K = match args.K {
    Some(K) => K,
    None => average_connectivity(gamma, args.N),
  };
  let network_config = NetworkConfig {
    N: args.N, K, gamma, seed: args.network_seed,
  };
  /* END: Network config. */

  /* BEGIN: Dynamics config. */
  let dynamics_config = DynamicsConfig {
    num_steps: args.num_steps,
    num_initial_conditions: args.num_initial_conditions,
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
   format!("experiment-{timestamp}.pb", timestamp=metadata.start_time.timestamp())
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