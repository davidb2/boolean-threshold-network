#ifndef NETWORK_NETWORK_H_
#define NETWORK_NETWORK_H_

#include <string>

namespace network {

struct NetworkConfig {
  int N;
  int k;
  int seed;
};


class BooleanThresholdNetwork {};

class Trajectories {};

void GetNetwork(const NetworkConfig&, BooleanThresholdNetwork*);

} // namespace network

#endif // NETWORK_NETWORK_H_
