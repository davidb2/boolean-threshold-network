use std::path::PathBuf;
use std::fs;
use chrono::Utc;
use clap::Parser;
use prost::Message;
use glob::glob;
use rayon::prelude::*;
use itertools::Itertools;
use boolean_threshold_network::types;
use csv::Writer;

pub mod pb {
  include!(concat!(env!("OUT_DIR"), "/boolean_threshold_network.rs"));
}

use types::State;
use pb::{
  Experiment as PBExperiment,
  ExperimentConfig as PBExperimentConfig,
  Network as PBNetwork,
  PowerLawDistribution as PBPowerLawDistribution,
  network_config::OutDegreeDistribution::PowerLawOutDegreeDistribution as PBPowerLawOutDegreeDistribution,
};

struct HammingDistanceRecord {
  pub actual_connectivity: f64,
  pub expected_connectivity: f64,
  pub hamming_distance: f64,
  pub gamma: Option<f64>,
}

fn write_hamming_distances(hamming_distances: &Vec<HammingDistanceRecord>, output_filename: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>  {
  let mut csv_writer = Writer::from_path(output_filename)?;
  _ = csv_writer.write_record(&["expected_connectivity", "actual_connectivity", "gamma", "hamming_distance"]);
  _ = hamming_distances.iter().for_each(|record|
    _ = csv_writer.write_record(&[
      record.expected_connectivity,
      record.actual_connectivity,
      record.gamma.unwrap_or_else(|| f64::NAN),
      record.hamming_distance,
    ].iter().map(|x| x.to_string()).collect::<Vec<String>>())
  );
  Ok(())
}
fn hamming_distance(state1: &State, state2: &State) -> f64 {
  state1
    .iter()
    .zip(state2)
    .map(|(a, b)| a ^ b)
    .filter(|dist| *dist)
    .count() as f64
  / (state1.len() as f64)
}

fn compute_actual_connectivity(network: &PBNetwork, experiment_config: &PBExperimentConfig) -> f64 {
  network
    .edges
    .iter()
    .counts_by(|edge| edge.source)
    .values()
    .sum::<usize>() as f64
    / (experiment_config.network_config.unwrap().network_size as f64)
}
fn compute_hamming_distances(experiments: &[PBExperiment]) -> Vec<HammingDistanceRecord> {
  let dynamics_config = experiments[0].experiment_config.unwrap().dynamics_config.unwrap();
  experiments
    .iter()
    .par_bridge()
    .map(|experiment|
      experiment.results
        .iter()
        .map(|result| {
          let original_network = match &result.original_network {
            None => panic!("No network"),
            Some(net) => net,
          };
          let actual_connectivity = compute_actual_connectivity(&original_network, &experiments[0].experiment_config.unwrap());
          (0..experiment.experiment_config.unwrap().dynamics_config.unwrap().num_initial_conditions)
            .combinations(2)
            .map(|initial_condition_idxs| HammingDistanceRecord {
              actual_connectivity,
              expected_connectivity: experiment.experiment_config.unwrap().network_config.unwrap().expected_connectivity,
              gamma: match experiment.experiment_config.unwrap().network_config.unwrap().out_degree_distribution.unwrap() {
                PBPowerLawOutDegreeDistribution(PBPowerLawDistribution { gamma }) => Some(gamma),
                _ => None,
              },
              hamming_distance: (0..dynamics_config.num_final_states_to_store)
                .map(|step| hamming_distance(
                  &result.perturbations[0].trajectories[initial_condition_idxs[0] as usize].states[step as usize].state,
                  &result.perturbations[0].trajectories[initial_condition_idxs[1] as usize].states[step as usize].state,
                )
                ).sum::<f64>() / (dynamics_config.num_final_states_to_store as f64),
            }
            )
            .collect::<Vec<HammingDistanceRecord>>()
        })
        .flatten()
        .collect::<Vec<HammingDistanceRecord>>()
    )
    .flatten()
    .collect::<Vec<HammingDistanceRecord>>()
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// the input directory of the experiment results
  #[arg(long)]
  input_directory: PathBuf,

  /// the output directory of the analyzed results
  #[arg(long)]
  output_directory: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
  let args = Args::parse();

  // Validate the args.
  println!("Validating inputs...");
  assert!(args.input_directory.exists(), "the provided input directory does not exist.");
  assert!(args.output_directory.exists(), "the provided output directory does not exist.");

  let start_time =  Utc::now();

  println!("input directory: {:?}", args.input_directory.to_str());
  let binding = args.input_directory.join("*.pb");
  let glob_pattern = binding.to_str().unwrap();
  println!("glob pattern: {:?}", glob_pattern);
  let experiments: Vec<PBExperiment> = glob(glob_pattern)?
    .par_bridge()
    .filter_map(|result| {println!("{:?}", result); result.ok()})
    .map(|f| {
      let protobuf_data = Vec::from(fs::read::<PathBuf>(f).expect("Unable to read file."));
      let mut buf = &protobuf_data[..];
      let experiment = PBExperiment::decode(&mut buf).unwrap();
      experiment
    })
    .collect();

  assert!(experiments.len() > 0, "Did not find any experiments");


  {
    let hamming_distances = compute_hamming_distances(&experiments);
    let output_path = args.output_directory.join(
      format!("hamming-distances-{timestamp}.csv", timestamp = start_time.timestamp_millis())
    );
    let output_filename = output_path.to_str().unwrap();
    _ = write_hamming_distances(&hamming_distances, output_filename);
  }

  Ok(())
}