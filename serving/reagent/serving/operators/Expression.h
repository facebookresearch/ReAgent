#pragma once

#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {
class Expression : public Operator {
 public:
  Expression(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~Expression() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs)
      override;

  virtual double runInternal(
      const std::string& equation,
      const StringDoubleMap& symbolTable);
};
} // namespace ml
