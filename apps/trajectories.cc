#include <cassert>
#include <fstream>
#include <iostream>

#include <gflags/gflags.h>

#include <boolean_network/dynamics.h>
#include <boolean_network/network.h>
#include <boolean_network/writer.h>

DEFINE_int32(N, 0, "N");
DEFINE_int32(k, 0, "k");
DEFINE_double(a, 0, "a");
DEFINE_int32(num_trials, 1, "num-trials");
DEFINE_int32(num_steps, 0, "num-steps");
DEFINE_uint32(seed, 0, "seed");
DEFINE_string(tag, "", "tag");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
    "Record the states of a boolean network with randomized initial starting states."
  );
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  assert(FLAGS_N > 0);
  assert(FLAGS_k > 0);
  assert(FLAGS_k <= FLAGS_N);
  assert(FLAGS_num_trials > 0);
  assert(FLAGS_num_steps > 0);
  assert(FLAGS_a > 0);

  network::NetworkConfig network_config;
  network_config.N = FLAGS_N;
  network_config.k = FLAGS_k;
  network_config.a = FLAGS_a;
  network_config.seed = FLAGS_seed;

  network::BooleanThresholdNetwork network(network_config);
  GetNetwork(network_config, &network);

  network::DynamicsConfig dynamics_config;
  dynamics_config.num_trials = FLAGS_num_trials;
  dynamics_config.num_steps = FLAGS_num_steps;

  network::MetaData metadata;
  metadata.tag = FLAGS_tag;
  metadata.start_time = std::chrono::system_clock::now();

  network::Trajectories trajectories;
  network::ComputeTrajectories(network, dynamics_config, &trajectories);

  std::cout << "done" << std::endl;
  metadata.end_time = std::chrono::system_clock::now();

  const std::string output_file_name = network::GetOutputFileName(
    "trajectories-",
    metadata.start_time
  );
  std::ofstream ofs{"data/" + output_file_name + ".json"};

  network::WriteTrajectoriesToStream(
    trajectories,
    network_config,
    dynamics_config,
    metadata,
    &ofs
  );
}