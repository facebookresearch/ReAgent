#pragma once

#include "reagent/serving/core/Headers.h"

namespace reagent {

class RealTimeCounter {
 public:
  virtual ~RealTimeCounter() = default;

  virtual int64_t getNumSamples(const std::string& key) = 0;

  virtual double getMean(const std::string& key) = 0;

  virtual double getVariance(const std::string& key) = 0;

  virtual void addValue(const std::string& key, double value) = 0;

  virtual void clear(const std::string& key) = 0;
};

} // namespace ml
