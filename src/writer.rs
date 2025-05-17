use prost::Message;

use crate::types::{
  Network,
  MetaData,
  ExperimentConfig,
  ExperimentResults,
};

pub mod pb {
  include!(concat!(env!("OUT_DIR"), "/boolean_threshold_network.rs"));
}

use pb::{
  Trajectory as PBTrajectory,
  MetaData as PBMetaData,
  DynamicsConfig as PBDynamicsConfig,
  NetworkConfig as PBNetworkConfig,
  Network as PBNetwork,
  Edge as PBEdge,
  Result as PBResult,
  State as PBState,
  ExperimentConfig as PBExperimentConfig,
  Perturbation as PBPerturbation,
  DrugConfig as PBDrugConfig,
  Experiment as PBExperiment,
  EdgePerturbation as PBEdgePerturbation,
};

fn to_pb_network(network: &Network) -> PBNetwork {
  PBNetwork {
    edges: network
      .get_representation_of_network()
      .iter()
      .map(|edge| PBEdge {
        source: edge.source as u32,
        target: edge.target as u32,
        weight: edge.weight
      })
      .collect()
  }
}

// TODO: For each network, write a Result protobuf per file (i.e. sharding). Put all results under
// the same directory. This would significantly reduce I/O time both in serialization and
// deserialization if we use threads to read/write to the files. The downside though is that
// the data storage format gets more complicated. For now, we leave it as is.
pub fn write_protobuf(
  experiment_results: &ExperimentResults,
  experiment_config: &ExperimentConfig,
  metadata: &MetaData,
  output_filename: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
  let pb = PBExperiment {
    metadata: Some(PBMetaData {
      tag: metadata.tag.clone(),
      start_time_s: metadata.start_time.timestamp() as u32,
      end_time_s: match metadata.end_time {
        None => None,
        Some(end_time) => Some(end_time.timestamp() as u32),
      }
    }),
    experiment_config: Some(PBExperimentConfig {
      num_networks: experiment_config.num_networks as u32,
      drug_config: Some(PBDrugConfig {
        num_drugs: experiment_config.drug_config.num_drugs as u32,
        num_targets_per_drug: experiment_config.drug_config.num_targets_per_drug as u32,
        drug_strength: experiment_config.drug_config.drug_strength,
        seed: experiment_config.drug_config.seed as u32
      }),
      dynamics_config: Some(PBDynamicsConfig {
        num_steps: experiment_config.dynamics_config.num_steps as u32,
        num_initial_conditions: experiment_config.dynamics_config.num_initial_conditions as u32,
        seed: experiment_config.dynamics_config.seed as u32
      }),
      network_config: Some(PBNetworkConfig {
        network_size: experiment_config.network_config.N as u32,
        expected_connectivity: experiment_config.network_config.K,
        gamma: experiment_config.network_config.gamma,
        seed: experiment_config.network_config.seed as u32
      })
    }),
    results: experiment_results
      .iter()
      .map(|experiment_result| PBResult {
          network_idx: experiment_result.network_idx as u32,
          perturbations: experiment_result
            .perturbations
            .iter()
            .map(|perturbation| PBPerturbation {
              name: perturbation.name.clone(),
              trajectories: perturbation
                .trajectories
                .iter()
                .map(|trajectory| PBTrajectory {
                  initial_condition_idx: trajectory.initial_condition_idx as u32,
                  states: trajectory
                    .states
                    .iter()
                    .enumerate()
                    .map(|(step_num, state)| PBState {
                      step_num: step_num as u32,
                      state: state.clone()
                    })
                    .collect()
                })
                .collect(),
              edge_perturbations: perturbation
                .edge_perturbations
                .iter()
                .map(|edge_perturbation| PBEdgePerturbation {
                  source: edge_perturbation.source as u32,
                  target: edge_perturbation.target as u32,
                  delta: edge_perturbation.delta,
                })
                .collect(),
            })
            .collect(),
          original_network: Some(to_pb_network(&experiment_result.original_network))
        })
      .collect()
  };

  let bytes = pb.encode_to_vec();
  std::fs::write(output_filename, &bytes)?;
  Ok(())
}