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
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~Ucb() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  std::string runInternal(
      const DecisionRequest& request,
      const std::string& method);

  virtual void giveFeedback(
      const Feedback& feedback,
      const StringOperatorDataMap& pastInputs,
      const OperatorData& pastOuptut) override;

  double getArmExpectation(const std::string& armName);
};
} // namespace ml
