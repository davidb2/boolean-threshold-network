use std::path::PathBuf;
use std::fs;
use chrono::Utc;
use clap::Parser;
use prost::Message;
use glob::glob;
use rayon::prelude::*;
use csv::Writer;

pub mod pb {
  include!(concat!(env!("OUT_DIR"), "/boolean_threshold_network.rs"));
}

use pb::{
  Experiment as PBExperiment,
  Edge as PBEdge,
};

struct StateRecord {
  initial_condition_idx: u32,
  network_idx: u32,
  drug_name: String,
  state: Vec<bool>,
}

struct NetworkRecord {
  network_idx: u32,
  edge: PBEdge,
}

fn write_states(
  states: &Vec<StateRecord>,
  num_features: u32,
  output_filename: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>  {
  let mut csv_writer = Writer::from_path(output_filename)?;
  _ = csv_writer.write_record(
    vec!["drug_name", "original_network_idx", "initial_condition_idx"]
      .iter()
      .map(|s| String::from(*s))
      .chain((0..num_features).map(|idx| format!("node-{:?}", idx)))
      .collect::<Vec<String>>()
  );
  _ = states.iter().for_each(|record|
    _ = csv_writer.write_record(
      vec![
        record.drug_name.clone(),
        record.network_idx.to_string(),
        record.initial_condition_idx.to_string(),
      ].into_iter()
       .chain(record.state.iter().map(|value| value.to_string()))
       .collect::<Vec<String>>()
    )
  );
  Ok(())
}

fn write_networks(
  networks: &Vec<NetworkRecord>,
  output_filename: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>  {
  let mut csv_writer = Writer::from_path(output_filename)?;
  _ = csv_writer.write_record(
    &["original_network_idx", "source", "target", "weight"]
  );
  _ = networks.iter().for_each(|record|
    _ = csv_writer.write_record(&[
      record.network_idx.to_string(),
      record.edge.source.to_string(),
      record.edge.target.to_string(),
      record.edge.weight.to_string(),
    ])
  );
  Ok(())
}

fn compute_networks(experiments: &Vec<PBExperiment>) -> Vec<NetworkRecord> {
  experiments.par_iter().flat_map(|experiment| {
    experiment.results.iter().map(|result| {
      result.original_network.clone().unwrap().edges.iter().map(|edge| {
        NetworkRecord {
          network_idx: result.network_idx.clone(),
          edge: edge.clone(),
        }
      }).collect::<Vec<NetworkRecord>>()
    }).flatten().collect::<Vec<NetworkRecord>>()
  }).collect::<Vec<NetworkRecord>>()
}

fn compute_states(experiments: &Vec<PBExperiment>) -> Vec<StateRecord> {
  experiments.par_iter().flat_map(|experiment| {
    experiment.results.iter().map(|result| {
      result.perturbations.iter().map(|perturbation| {
        perturbation.trajectories.iter().map(|trajectory| {
          trajectory.states.iter().map(|state| {
            StateRecord {
              initial_condition_idx: trajectory.initial_condition_idx,
              drug_name: perturbation.name.clone(),
              network_idx: result.network_idx,
              state: state.state.clone(),
            }
          }).collect::<Vec<StateRecord>>()
        }).flatten().collect::<Vec<StateRecord>>()
      }).flatten().collect::<Vec<StateRecord>>()
    }).flatten().collect::<Vec<StateRecord>>()
  }).collect::<Vec<StateRecord>>()
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


  let timestamp = start_time.timestamp_millis();
  {
    println!("Extracting states");
    let states = compute_states(&experiments);
    let output_path = args.output_directory.join(
      format!("states-{timestamp}.csv", timestamp=timestamp)
    );
    let output_filename = output_path.to_str().unwrap();
    println!("Writing states");
    _ = write_states(
      &states,
      experiments[0].experiment_config.unwrap().network_config.unwrap().network_size,
      output_filename
    );
  }

  {
    println!("Extracting networks");
    let networks = compute_networks(&experiments);
    let output_path = args.output_directory.join(
      format!("networks-{timestamp}.csv", timestamp=timestamp)
    );
    let output_filename = output_path.to_str().unwrap();
    println!("Writing networks");
    _ = write_networks(&networks, output_filename);
  }

  Ok(())
}