#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class Ucb : public Operator {
 public:
  Ucb(const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);
  }

  virtual ~Ucb() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  RankedActionList runInternal(
      const DecisionRequest& request,
      const std::string& method);

  virtual void giveFeedback(
      const Feedback& feedback,
      const StringOperatorDataMap& pastInputs,
      const OperatorData& pastOuptut) override;

  double getArmExpectation(const std::string& armName);

 protected:
  std::mt19937 generator_;
};
} // namespace reagent
