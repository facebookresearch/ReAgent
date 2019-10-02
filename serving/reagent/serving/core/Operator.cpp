#include "reagent/serving/core/Operator.h"

#include "reagent/serving/core/DecisionService.h"

namespace reagent {
Operator::Operator(
    const std::string& name,
    const std::string& planName,
    const StringStringMap& inputDepMap,
    const DecisionService* const decisionService)
    : name_(name),
      planName_(planName),
      inputDepMap_(inputDepMap),
      actionValueScorer_(decisionService->getActionValueScorer()),
      logJoiner_(decisionService->getLogJoiner()),
      realTimeCounter_(decisionService->getRealTimeCounter()),
      sharedParameterHandler_(decisionService->getSharedParameterHandler()) {
  for (const auto& it : inputDepMap) {
    deps_.insert(it.second);
  }
}
} // namespace reagent
