#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <set>

#include <boolean_network/network.h>

namespace network {

/// Generates random threshold boolean network.
BooleanThresholdNetwork::BooleanThresholdNetwork(const NetworkConfig& config)
  : N_(config.N),
    gen_(config.seed),
    out_weights_(config.N, config.N),
    thresholds_(config.N, 0),
    coin_flip_dist_(.5) {
  GenerateWeights_(config);
  GenerateThresholds_(config);
}

State BooleanThresholdNetwork::GetUniformlyRandomState() {
  State state(N_, false);
  for (size_t i = 0; i < N_; ++i) {
    state[i] = coin_flip_dist_(gen_);
  }
  return state;
}

double BooleanThresholdNetwork::GetRandomEdgeWeight_() {
  return 2 * static_cast<double>(coin_flip_dist_(gen_)) - 1;
}

void BooleanThresholdNetwork::GenerateThresholds_(const NetworkHyperParameters& params) {
  thresholds_.resize(N_);
  // This might be superfluous.
  std::fill_n(thresholds_.begin(), N_, 0.);
}

void BooleanThresholdNetwork::GenerateWeights_(const NetworkHyperParameters& params) {
  const auto node_weights = GenerateNodeWeights_(params);

  std::vector<Eigen::Triplet<int>> triplets;
  for (int v = 0; v < N_; ++v) {
    const auto incoming_nodes = SampleNodes_(node_weights, params);
    for (const auto u : incoming_nodes) {
      const auto edge_weight = GetRandomEdgeWeight_();
      // Add edge u -> v with weight edge_weight.
      triplets.emplace_back(u, v, edge_weight);
    }
  }

  out_weights_.setFromTriplets(triplets.begin(), triplets.end());
}

bool Activation(double incoming_weight, double threshold, bool previous_state) {
  return (incoming_weight == threshold) ? previous_state : (incoming_weight > threshold);
}

std::vector<double> BooleanThresholdNetwork::GenerateNodeWeights_(const NetworkHyperParameters& params) {
  std::vector<double> weights(N_);
  for (int i = 0; i < N_; ++i) {
    weights[i] = std::pow(i + 1, -params.a);
  }
  return weights;
}

std::set<int> BooleanThresholdNetwork::SampleNodes_(
  const std::vector<double>& weights,
  const NetworkHyperParameters& params
) {
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  std::set<int> sampled_nodes;
  for (int i = 0; i < params.k; ++i) {
    sampled_nodes.insert(dist(gen_));
  }
  return sampled_nodes;
}

State BooleanThresholdNetwork::GetNextState(const State& previous_state) const {
  assert(previous_state.size() == N_);

  State next_state(N_, false);
  for (int j = 0; j < N_; ++j) {
    double sum = 0;
    for (WeightMatrix ::InnerIterator it(out_weights_, j); it; ++it) {
      const auto i = it.row();
      const auto W_ij = it.value();
      sum += W_ij * static_cast<double>(previous_state[i]);
    }
    next_state[j] = Activation(
      sum,
      thresholds_[j],
      previous_state[j]
    );
  }

  return next_state;
}

}  // namespace network