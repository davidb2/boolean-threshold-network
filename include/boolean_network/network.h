#ifndef NETWORK_NETWORK_H_
#define NETWORK_NETWORK_H_

#include <random>
#include <set>
#include <string>
#include <vector>

#include <Eigen/SparseCore>

namespace network {

using State = std::vector<bool>;
using WeightMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

struct NetworkHyperParameters {
  double a;
  int k;
};

struct NetworkConfig : NetworkHyperParameters {
  int N;
  unsigned int seed;
};


class BooleanThresholdNetwork {
 public:
  BooleanThresholdNetwork(const NetworkConfig&);
  State GetNextState(const State&) const;
  State GetUniformlyRandomState();

 private:
  double GetRandomEdgeWeight_();
  void GenerateWeights_(const NetworkHyperParameters&);
  void GenerateThresholds_(const NetworkHyperParameters&);
  std::vector<double> GenerateNodeWeights_(const NetworkHyperParameters&);
  std::set<int> SampleNodes_(
    const std::vector<double>& weights,
    const NetworkHyperParameters& params
  );

 private:
  int N_;
  std::mt19937 gen_;
  WeightMatrix out_weights_;
  std::vector<double> thresholds_;
  std::bernoulli_distribution coin_flip_dist_;
};

class Trajectories {};

void GetNetwork(const NetworkConfig&, BooleanThresholdNetwork*);

} // namespace network

#endif // NETWORK_NETWORK_H_
