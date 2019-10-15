#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class DecisionPlan {
 public:
  DecisionPlan(const DecisionConfig& config,
               const std::vector<std::shared_ptr<Operator>>& operators,
               const StringOperatorDataMap& constants);

  const DecisionConfig& getConfig() { return config_; }

  const std::vector<std::shared_ptr<Operator>>& getOperators() {
    return operators_;
  }

  const StringOperatorDataMap& getConstants() { return constants_; }

  const std::string& getOutputOperatorName() {
    if (operators_.empty()) {
      LOG_AND_THROW("Tried to get output operator name but no operators exist");
    }
    return operators_.back()->getName();
  }

  double evaluateRewardFunction(const StringDoubleMap& metrics);

 protected:
  DecisionConfig config_;
  std::vector<std::shared_ptr<Operator>> operators_;
  StringOperatorDataMap constants_;
};
}  // namespace reagent
