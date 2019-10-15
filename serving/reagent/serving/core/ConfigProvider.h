#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/DecisionPlan.h"

namespace reagent {
class DecisionService;

class ConfigProvider {
 public:
  ConfigProvider() {}

  virtual void initialize(DecisionService* decisionService) {
    decisionService_ = decisionService;
  }

  virtual ~ConfigProvider() = default;

 protected:
  DecisionService* decisionService_;
};
} // namespace reagent
