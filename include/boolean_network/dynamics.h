#ifndef NETWORK_DYNAMICS_H_
#define NETWORK_DYNAMICS_H_

#include "network.h"

namespace network {

struct DynamicsConfig {
  int num_trials;
  int seed;
};

void ComputeTrajectories(
  const BooleanThresholdNetwork&,
  const DynamicsConfig&,
  Trajectories*
);

}  // namespace network

#endif // NETWORK_DYNAMICS_H_
