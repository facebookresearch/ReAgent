#pragma once

#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {
class EpsilonGreedyRanker : public Operator {
 public:
  EpsilonGreedyRanker(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);
  }

  virtual ~EpsilonGreedyRanker() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  virtual RankedActionList runInternal(
      const StringDoubleMap& input,
      double epsilon);

 protected:
  std::mt19937 generator_;
};
} // namespace reagent
