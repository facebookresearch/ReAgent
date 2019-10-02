#pragma once

#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {
class Frechet : public Operator {
 public:
  Frechet(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~Frechet() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs)
      override;

  virtual StringDoubleMap
  run(const StringDoubleMap& input, double rho, double gamma, int seed);
};
} // namespace ml
