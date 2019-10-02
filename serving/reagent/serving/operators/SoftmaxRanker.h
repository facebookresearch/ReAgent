#pragma once

#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {
class SoftmaxRanker : public Operator {
 public:
  SoftmaxRanker(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~SoftmaxRanker() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs)
      override;

  virtual StringList run(const StringDoubleMap& input, double temperature, int seed);
};
} // namespace ml
