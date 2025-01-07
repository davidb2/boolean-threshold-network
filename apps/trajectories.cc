#include <cassert>
#include <iostream>

#include <gflags/gflags.h>

#include <network/dynamics.h>
#include <network/network.h>

DEFINE_int32(N, 0, "N");
DEFINE_int32(k, 0, "k");
DEFINE_int32(num_trials, 1, "num-trials");
DEFINE_int32(seed, 0, "seed");
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

  network::DynamicsConfig config;
  config.N = FLAGS_N;
  config.k = FLAGS_k;
  config.num_trials = FLAGS_num_trials;
  config.seed = FLAGS_seed;

  network::BooleanThresholdNetwork network;
  GetNetwork(config, &network);

  network::MetaData metadata;
  metadata.tag = FLAGS_tag;
  metadata.start_time = std::chrono::system_clock::now();

  network::Trajectories trajectories;
  network::ComputeTrajectories(config, &trajectories);

  std::cout << "done" << std::endl;
  metadata.end_time = std::chrono::system_clock::now();

  const std::string output_file_name = network::GetOutputFileName("trajectories-", metadata.start_time);
  std::ofstream ofs{"data/" + output_file_name + ".json"};

  network::WriteTrajectoriesToStream(trajectories, config, metadata, &ofs);
}