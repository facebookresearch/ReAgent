#pragma once

#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {
class InputFromRequest : public Operator {
 public:
  InputFromRequest(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~InputFromRequest() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs)
      override;
};
} // namespace ml
