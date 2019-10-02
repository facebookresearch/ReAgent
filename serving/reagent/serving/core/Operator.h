#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/ActionValueScorer.h"
#include "reagent/serving/core/LogJoiner.h"
#include "reagent/serving/core/RealTimeCounter.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
class DecisionService;

class Operator {
 public:
  Operator(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService);

  virtual ~Operator() = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) = 0;

  virtual void giveFeedback(
      const Feedback&,
      const StringOperatorDataMap& /* pastInputs */,
      const OperatorData& /* pastOuptut */) {}

  inline const std::set<std::string>& getDeps() {
    return deps_;
  }

  inline const StringStringMap& getInputDepMap() {
    return inputDepMap_;
  }

  inline const std::string& getName() {
    return name_;
  }

  inline static StringList getActionNamesFromRequest(
      const DecisionRequest& request) {
    bool discreteActions = !request.actions.names.empty();

    StringList actionNames;
    if (discreteActions) {
      actionNames = request.actions.names;
    } else {
      for (const auto& it : request.actions.features) {
        actionNames.emplace_back(it.first);
      }
    }

    return actionNames;
  }

 protected:
  std::string name_;
  std::string planName_;
  std::set<std::string> deps_;
  StringStringMap inputDepMap_;

  std::shared_ptr<ActionValueScorer> actionValueScorer_;
  std::shared_ptr<LogJoiner> logJoiner_;
  std::shared_ptr<RealTimeCounter> realTimeCounter_;
  std::shared_ptr<SharedParameterHandler> sharedParameterHandler_;

  Operator(const Operator&) = delete; // non construction-copyable
  Operator& operator=(const Operator&) = delete; // non copyable
};
} // namespace reagent
