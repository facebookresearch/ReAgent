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
      const DecisionService* const decisionService);

  virtual ~SoftmaxRanker() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  virtual RankedActionList run(
      const StringDoubleMap& input,
      double temperature);

 protected:
  std::mt19937 generator_;
};
} // namespace reagent
