#pragma once

#include "reagent/serving/core/RealTimeCounter.h"

namespace reagent {

class LocalRealTimeCounter : public RealTimeCounter {
 public:
  LocalRealTimeCounter() : windowSize_(1024 * 1024) {}

  virtual ~LocalRealTimeCounter() override = default;

  virtual int64_t getNumSamples(const std::string& key) override;

  virtual double getMean(const std::string& key) override;

  virtual double getVariance(const std::string& key) override;

  virtual void addValue(const std::string& key, double value) override;

  void setWindowSize(int windowSize) {
    windowSize_ = windowSize;
  }

  virtual void clear(const std::string& key) override;

 protected:
  std::unordered_map<std::string, std::deque<double>> counts_;
  int windowSize_;
};

} // namespace reagent
