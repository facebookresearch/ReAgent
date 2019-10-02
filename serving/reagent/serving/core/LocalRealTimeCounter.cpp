#include "reagent/serving/core/LocalRealTimeCounter.h"

namespace reagent {

int64_t LocalRealTimeCounter::getNumSamples(const std::string& key) {
  auto it = counts_.find(key);
  if (it == counts_.end()) {
    return 0;
  }
  return it->second.size();
}

double LocalRealTimeCounter::getMean(const std::string& key) {
  auto it = counts_.find(key);
  if (it == counts_.end()) {
    return 0;
  }
  auto& v = it->second;
  double sum = 0;
  for (auto i : v) {
    sum += i;
  }
  return sum / v.size();
}

double LocalRealTimeCounter::getVariance(const std::string& key) {
  auto it = counts_.find(key);
  if (it == counts_.end()) {
    return 0;
  }
  auto& v = it->second;
  double mean = getMean(key);
  double varSum = 0;
  for (auto i : v) {
    varSum += (i - mean) * (i - mean);
  }
  return varSum / v.size();
}

void LocalRealTimeCounter::addValue(const std::string& key, double value) {
  auto it = counts_.find(key);
  if (it == counts_.end()) {
    counts_[key] = std::deque<double>();
    it = counts_.find(key);
  }
  it->second.push_back(value);
  while (int(it->second.size()) > windowSize_) {
    it->second.pop_front();
  }
}

} // namespace reagent
