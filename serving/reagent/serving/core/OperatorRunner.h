#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class OperatorRunner {
 public:
  OperatorRunner() {}

  StringOperatorDataMap run(const std::vector<std::shared_ptr<Operator>>& ops,
                            const StringOperatorDataMap& constants,
                            const DecisionRequest& request,
                            const OperatorData& extraInput);

 protected:
  tf::Executor taskExecutor_;
};
}  // namespace reagent
