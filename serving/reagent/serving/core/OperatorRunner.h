#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

namespace reagent {
class OperatorRunner {
 public:
  OperatorRunner() {
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(8);
  }

  StringOperatorDataMap run(
      const std::vector<std::shared_ptr<Operator>>& ops,
      const StringOperatorDataMap& constants,
      const DecisionRequest& request,
      const OperatorData& extraInput);

 protected:
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
};
} // namespace ml
