#ifndef NETWORK_WRITER_H_
#define NETWORK_WRITER_H_

#include <chrono>
#include <fstream>
#include <string>

#include "dynamics.h"
#include "network.h"

namespace network {

struct MetaData {
  std::string tag;
  std::chrono::time_point<std::chrono::system_clock> start_time;
  std::chrono::time_point<std::chrono::system_clock> end_time;
};

std::string GetOutputFileName(
  const std::string& prefix,
  const std::chrono::time_point<std::chrono::system_clock>& time
);

void WriteTrajectoriesToStream(
  const Trajectories&,
  const NetworkConfig&,
  const DynamicsConfig&,
  const MetaData&,
  std::ofstream*
);

} // namespace network

#endif // NETWORK_WRITER_H_
