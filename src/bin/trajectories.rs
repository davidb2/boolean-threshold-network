use clap::Parser;
use chrono::Utc;

// Bring in your library’s types & functions:
use boolean_threshold_network::{
  types::NetworkConfig,
  types::DynamicsConfig,
  types::BooleanThresholdNetwork,
  types::MetaData,
  dynamics::compute_trajectories,
  writer::write_protobuf,
  // GetOutputFileName,
  // write_trajectories_to_stream,
};

// use generated::boolean_threshold_network::;
pub mod pb {
  include!(concat!(env!("OUT_DIR"), "/boolean_threshold_network.rs"));
}

/// Record the states of a boolean network with randomized initial starting states.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Number of nodes in the network
  #[arg(short = 'N', long)]
  N: usize,

  /// Threshold parameter
  #[arg(long)]
  gamma: f64,

  /// How many independent trials to run
  #[arg(long, default_value_t = 1)]
  num_trials: usize,

  /// How many steps per trial
  #[arg(long)]
  num_steps: usize,

  /// RNG seed for the network itself
  #[arg(long, default_value_t = 0)]
  network_seed: usize,

  /// RNG seed for the dynamics
  #[arg(long, default_value_t = 0)]
  dynamics_seed: usize,

  /// A tag to stick in the metadata
  #[arg(long, default_value = "")]
  tag: String,
}


fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
  let args = Args::parse();

  assert!(args.N > 0, "N must be > 0");
  assert!(args.num_trials > 0, "num_trials must be > 0");
  assert!(args.num_steps > 0, "num_steps must be > 0");

  let network_config = NetworkConfig {
    N: args.N,
    gamma: args.gamma,
    seed: args.network_seed,
  };
  let dynamics_config = DynamicsConfig {
    num_trials: args.num_trials,
    num_steps: args.num_steps,
    seed: args.dynamics_seed,
  };

  let mut network = BooleanThresholdNetwork::new(&network_config);

  let mut metadata = MetaData {
    tag: args.tag.clone(),
    start_time: Utc::now(),
    end_time: Utc::now(),
  };

  let trajectories = compute_trajectories(&mut network, &dynamics_config);
  metadata.end_time = Utc::now();

  write_protobuf(
    "data/out.trajpb",
    &trajectories,
    &metadata,
    &dynamics_config,
    &network_config,
    &network,
  )?;
  println!("Wrote {}", "out.trajpb");
  println!("done");

  // let fname = GetOutputFileName("trajectories-", metadata.start_time);
  // let path = format!("data/{}.json", fname);

  // let file = File::create(&path)?;
  // let mut writer = BufWriter::new(file);
  // write_trajectories_to_stream(
  //   &trajectories,
  //   &network,
  //   &network_config,
  //   &dynamics_config,
  //   &metadata,
  //   &mut writer,
  // )?;

  Ok(())
}
