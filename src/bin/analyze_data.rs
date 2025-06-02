use std::collections::HashMap;
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
  Trajectory as PBTrajectory,
  ExperimentConfig as PBExperimentConfig,
};

struct HammingDistanceRecord {
  pub expected_connectivity: f64,
  pub hamming_distance: f64,
}

struct MutualInfoRecord {
  pub expected_connectivity: f64,
  pub mutual_info: f64,
}

fn write_hamming_distances(hamming_distances: &Vec<HammingDistanceRecord>, output_filename: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>  {
  let mut csv_writer = Writer::from_path(output_filename)?;
  _ = csv_writer.write_record(&["expected_connectivity", "hamming_distance"]);
  _ = hamming_distances.iter().for_each(|record|
    _ = csv_writer.write_record(&[record.expected_connectivity, record.hamming_distance].iter().map(|x| x.to_string()).collect::<Vec<String>>())
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

fn write_mutual_infos(mutual_infos: &Vec<MutualInfoRecord>, output_filename: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>  {
  let mut csv_writer = Writer::from_path(output_filename)?;
  _ = csv_writer.write_record(&["expected_connectivity", "mutual_info"]);
  _ = mutual_infos.iter().for_each(|record|
    _ = csv_writer.write_record(&[record.expected_connectivity, record.mutual_info].iter().map(|x| x.to_string()).collect::<Vec<String>>())
  );
  Ok(())
}

fn mutual_info(trajectories: &[PBTrajectory], time: usize, experiment_config: &PBExperimentConfig) -> f64 {
  let joint_counts: Vec<HashMap<(bool, bool), usize>> =
    (0..experiment_config.network_config.unwrap().network_size)
      .combinations(2)
      .map(|node_idxs|
        trajectories
          .iter()
          .map(|trajectory| (
            trajectory.states[time].state[node_idxs[0] as usize],
            trajectory.states[time].state[node_idxs[1] as usize],
          ))
          .counts(),
      )
      .collect();

  let joint_probs: Vec<HashMap<(bool, bool), f64>> = joint_counts
    .iter()
    .map(|joint_count| joint_count
      .iter()
      .map(|((x, y), count)|
        ((*x, *y), (*count as f64) / (joint_count.values().sum::<usize>() as f64))
      ).collect()
    )
    .collect();

  let pss: Vec<HashMap<(Option<bool>, Option<bool>), f64>>  = joint_probs
    .iter()
    .map(|joint_prob| {
      let closure = |x: &Option<bool>, y: &Option<bool>| -> f64 {
        match (x, y) {
          (None, None) => 1.,
          (Some(x), None) => vec![false, true].iter().map(|z| match joint_prob.get(&(*x, *z)) {
            None => 0.,
            Some(value) => *value,
          }).sum(),
          (None, Some(y)) => vec![false, true].iter().map(|z| match joint_prob.get(&(*z, *y)) {
            None => 0.,
            Some(value) => *value,
          }).sum(),
          (Some(x), Some(y)) => match joint_prob.get(&(*x, *y)) {
            None => 0.,
            Some(value) => *value,
          },
        }
      };
      vec![None, Some(false), Some(true)]
        .iter()
        .cartesian_product( &vec![None, Some(false), Some(true)])
        .map(|(x, y)| ((*x, *y), closure(x, y)))
        .collect()
    }
    )
  .collect();

  let mutual_infos: Vec<f64> = pss
    .iter()
    .map(|ps| {
      vec![Some(false), Some(true)]
        .iter()
        .cartesian_product( &vec![Some(false), Some(true)])
        .map(|(x, y)|
          match (*ps.get(&(*x, *y)).unwrap(), *ps.get(&(*x, None)).unwrap(), *ps.get(&(None, *y)).unwrap()) {
            (0., _, _) | (_, 0., _) | (_, _, 0.) => 0.,
            (ps_xy, ps_x, ps_y) => ps_xy * f64::ln(ps_xy / (ps_x * ps_y)),
          }
        )
        .sum()
    })
    .collect();

  mutual_infos.iter().sum::<f64>() / (mutual_infos.len() as f64)
}

fn compute_hamming_distances(experiments: &[PBExperiment]) -> Vec<HammingDistanceRecord> {
  let last_step = experiments[0].experiment_config.unwrap().dynamics_config.unwrap().num_steps;
  experiments
    .iter()
    .par_bridge()
    .map(|experiment|
      experiment.results
        .iter()
        .map(|result|
          (0..experiment.experiment_config.unwrap().dynamics_config.unwrap().num_initial_conditions)
            .combinations(2)
            .map(|initial_condition_idxs| HammingDistanceRecord {
              expected_connectivity: experiment.experiment_config.unwrap().network_config.unwrap().expected_connectivity,
              hamming_distance: hamming_distance(
                &result.perturbations[0].trajectories[initial_condition_idxs[0] as usize].states[last_step as usize].state,
                &result.perturbations[0].trajectories[initial_condition_idxs[1] as usize].states[last_step as usize].state,
              )
            }
            )
            .collect::<Vec<HammingDistanceRecord>>()
        )
        .flatten()
        .collect::<Vec<HammingDistanceRecord>>()
    )
    .flatten()
    .collect::<Vec<HammingDistanceRecord>>()
}

fn compute_mutual_infos(experiments: &[PBExperiment]) -> Vec<MutualInfoRecord> {
  let last_step = experiments[0].experiment_config.unwrap().dynamics_config.unwrap().num_steps;
  experiments
    .iter()
    .par_bridge()
    .map(|experiment|
      experiment.results
        .iter()
        .map(|result| MutualInfoRecord {
          expected_connectivity: experiment.experiment_config.unwrap().network_config.unwrap().expected_connectivity,
          mutual_info: mutual_info(
            result.perturbations[0].trajectories.as_slice(),
            last_step as usize,
            &experiments[0].experiment_config.unwrap(),
          ),
        })
        .collect::<Vec<MutualInfoRecord>>()
    )
    .flatten()
    .collect::<Vec<MutualInfoRecord>>()
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
      format!("hamming-distances-{timestamp}.csv", timestamp = start_time.timestamp())
    );
    let output_filename = output_path.to_str().unwrap();
    _ = write_hamming_distances(&hamming_distances, output_filename);
  }

  {
    let mutual_infos = compute_mutual_infos(&experiments);
    let output_path = args.output_directory.join(
      format!("mutual-infos-{timestamp}.csv", timestamp = start_time.timestamp())
    );
    let output_filename = output_path.to_str().unwrap();
    _ = write_mutual_infos(&mutual_infos, output_filename);
  }


  Ok(())
}