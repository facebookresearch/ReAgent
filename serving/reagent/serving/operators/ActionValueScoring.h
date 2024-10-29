#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class ActionValueScoring : public Operator {
 public:
  ActionValueScoring(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~ActionValueScoring() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  StringDoubleMap
  runInternal(int modelId, int snapshotId, const DecisionRequest& request);
};
} // namespace reagent
