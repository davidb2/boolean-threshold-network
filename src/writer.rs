use chrono::Utc;
use prost::Message;
use sprs::CsMat;

use crate::types::{
  BooleanThresholdNetwork, DynamicsConfig, MetaData, NetworkConfig,
  State, Trajectories,
};

pub mod pb {
  include!(concat!(env!("OUT_DIR"), "/boolean_threshold_network.rs"));
}

use pb::{
  Trajectories as PBTraj, MetaData as PBMeta, DynamicsConfig as PBDyn, NetworkConfig as PBNetCfg,
  Network as PBNet, Edge as PBEdge, Result as PBResult, State as PBState,
};

pub fn write_protobuf(path: &str, trajs: &Trajectories, md: &MetaData, dcfg: &DynamicsConfig, ncfg: &NetworkConfig, net: &BooleanThresholdNetwork)
                  -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
  // build the protobuf struct
  let pb = trajs.to_pb_traj(md, dcfg, ncfg, net);

  // serialize to bytes
  let bytes = pb.encode_to_vec();

  // write file
  std::fs::write(path, &bytes)?;
  Ok(())
}

/// Convert your Rust MetaData → protobuf MetaData
impl From<&MetaData> for PBMeta {
  fn from(md: &MetaData) -> PBMeta {
    PBMeta {
      end_time_s:   md.end_time.timestamp() as u32,
      start_time_s: md.start_time.timestamp() as u32,
      tag:          md.tag.clone(),
    }
  }
}

/// Rust DynamicsConfig → protobuf DynamicsConfig
impl From<&DynamicsConfig> for PBDyn {
  fn from(c: &DynamicsConfig) -> PBDyn {
    PBDyn {
      num_steps:  c.num_steps as u32,
      num_trials: c.num_trials as u32,
      seed:       c.seed as u32,
    }
  }
}

/// Rust NetworkConfig → protobuf NetworkConfig
impl From<&NetworkConfig> for PBNetCfg {
  fn from(c: &NetworkConfig) -> PBNetCfg {
    PBNetCfg {
      n:     c.N as u32,
      gamma: c.gamma,
      seed:  c.seed as u32,
    }
  }
}

/// Rust BooleanThresholdNetwork → protobuf Network
impl From<&BooleanThresholdNetwork> for PBNet {
  fn from(net: &BooleanThresholdNetwork) -> PBNet {
    // Flatten CSC to edge list:
    let mut edges = Vec::new();
    for (col, col_vec) in net.out_weights.outer_iterator().enumerate() {
      for (row, &w) in col_vec.iter() {
        edges.push(PBEdge {
          from:   row as u32,
          to:     col as u32,
          weight: w,
        });
      }
    }
    PBNet {
      edges,
      thresholds: net.thresholds.clone(),
    }
  }
}

pub trait ToProtoState {
  fn to_pb_state(&self, step: usize) -> pb::State;
}

pub trait ToProtoTrajectories {
  fn to_pb_traj(
    &self,
    metadata: &MetaData,
    dyn_cfg: &DynamicsConfig,
    net_cfg: &NetworkConfig,
    net: &BooleanThresholdNetwork,
  ) -> pb::Trajectories;
}
impl ToProtoState for State {
  fn to_pb_state(&self, step: usize) -> pb::State {
    pb::State {
      state:    self.clone(),      // Vec<bool> → repeated bool
      step_num: step as u32,
    }
  }
}

impl ToProtoTrajectories for Trajectories {
  fn to_pb_traj(
    &self,
    metadata: &MetaData,
    dyn_cfg: &DynamicsConfig,
    net_cfg: &NetworkConfig,
    net: &BooleanThresholdNetwork,
  ) -> pb::Trajectories {
    //  a) Convert metadata, configs, network
    let md_pb     = pb::MetaData {
      end_time_s:   metadata.end_time.timestamp()  as u32,
      start_time_s: metadata.start_time.timestamp() as u32,
      tag:          metadata.tag.clone(),
    };
    let dyn_pb    = pb::DynamicsConfig {
      num_steps:  dyn_cfg.num_steps  as u32,
      num_trials: dyn_cfg.num_trials as u32,
      seed:       dyn_cfg.seed       as u32,
    };
    let netcfg_pb = pb::NetworkConfig {
      n:     net_cfg.N     as u32,
      gamma: net_cfg.gamma,
      seed:  net_cfg.seed  as u32,
    };
    //  b) Flatten your BooleanThresholdNetwork → pb::Network
    let network_pb = net.into();
    //  c) Convert each trial in self (Vec<Vec<State>>)
    let results_pb = self.iter()
      .enumerate()
      .map(|(trial_idx, states)| {
        let states_pb = states.iter()
          .enumerate()
          .map(|(step, st)| st.to_pb_state(step))
          .collect();
        pb::Result {
          states: states_pb,
          trial:  vec![trial_idx as u32],
        }
      })
      .collect();

    //  d) Assemble the top‐level message
    pb::Trajectories {
      metadata:        Some(md_pb),
      dynamics_config: Some(dyn_pb),
      network_config:  Some(netcfg_pb),
      network:         Some(network_pb),
      results:         results_pb,
    }
  }
}